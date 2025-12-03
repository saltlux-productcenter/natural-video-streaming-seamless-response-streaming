# sync_manager.py (완전 재작성)
import asyncio
import logging
import time
import os
from typing import Dict, Optional, List, Callable, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SentenceData:
    """문장별 데이터"""
    sentence_num: int
    tts_response: Optional[object] = None
    frames: List[str] = field(default_factory=list)
    is_complete: bool = False
    audio_duration: float = 0.0
    is_last_sentence: bool = False


class SyncManager:
    """TTS-Lipsync 실시간 동기화 (25 FPS)"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.lock = asyncio.Lock()

        # 문장 데이터
        self.sentences: Dict[int, SentenceData] = {}

        # 재생 상태
        self.current_sentence_num: Optional[int] = None
        self.current_frame_index = 0
        self.playback_start_time: Optional[float] = None
        self.is_playing = False
        self.audio_sent = False

        # 콜백
        self.audio_callback: Optional[Callable] = None
        self.video_callback: Optional[Callable] = None
        self.idle_callback: Optional[Callable] = None

        # 동기화 타이머
        self.sync_timer: Optional[asyncio.Task] = None
        self.frame_rate = 25
        self.frame_interval = 1.0 / 25

        # 턴 관리
        self.current_turn: int = 0
        self.is_idle: bool = True
        self.total_sentences: int = 0
        self.llm_completed: bool = False
        self.pending_total_sentences: int = 0

        logger.info(f"[SYNC] 세션 {session_id[:8]}: 실시간 스트리밍 매니저 생성 (25 FPS)")

    def _analyze_audio_duration(self, wav_data: bytes) -> float:
        """WAV 오디오 길이 계산"""
        try:
            if len(wav_data) < 44:
                return 0.0

            sample_rate = int.from_bytes(wav_data[24:28], byteorder='little')
            channels = int.from_bytes(wav_data[22:24], byteorder='little')
            bits_per_sample = int.from_bytes(wav_data[34:36], byteorder='little')

            audio_data_size = len(wav_data) - 44
            total_samples = audio_data_size // (channels * bits_per_sample // 8)
            duration = total_samples / sample_rate

            return round(duration, 3)
        except Exception as e:
            logger.error(f"[SYNC] 오디오 분석 실패: {e}")
            return 0.0

    def set_callbacks(self, audio_callback: Callable, video_callback: Callable, idle_callback: Callable):
        """콜백 함수 설정"""
        self.audio_callback = audio_callback
        self.video_callback = video_callback
        self.idle_callback = idle_callback

    def set_total_sentences(self, total: int):
        """전체 문장 수 설정 (LLM 완료 시 호출)"""
        # is_idle 상태면 다음 턴을 위한 예약
        if self.is_idle:
            self.pending_total_sentences = total
            logger.info(f"[SYNC] 턴#{self.current_turn}: 전체 문장 수 예약 = {total} (다음 턴에 적용)")
        else:
            # 재생 중이면 현재 턴에 즉시 적용
            self.total_sentences = total
            self.llm_completed = True
            logger.info(f"[SYNC] 턴#{self.current_turn}: 전체 문장 수 = {total}, LLM 완료")

            # 이미 완료된 문장의 is_last_sentence 업데이트
            if total > 0:
                for sentence_num, sentence in self.sentences.items():
                    if sentence.is_complete and sentence_num == total:
                        sentence.is_last_sentence = True
                        logger.info(f"[SYNC] 문장#{sentence_num}을 마지막 문장으로 업데이트")

    async def add_tts_response(self, sentence_num: int, tts_response) -> bool:
        """TTS 응답 추가"""
        async with self.lock:
            # 새 턴 시작 감지
            if sentence_num == 1 and self.is_idle:
                await self._start_new_turn()

            self.is_idle = False

            audio_duration = self._analyze_audio_duration(tts_response.audio_data)
            expected_frames = int(audio_duration * 25)

            logger.info(f"[SYNC] 턴#{self.current_turn} 문장#{sentence_num} TTS 추가 "
                        f"({audio_duration:.2f}초, {expected_frames}프레임 예상)")

            if sentence_num not in self.sentences:
                self.sentences[sentence_num] = SentenceData(
                    sentence_num=sentence_num,
                    tts_response=tts_response,
                    audio_duration=audio_duration
                )
            else:
                self.sentences[sentence_num].tts_response = tts_response
                self.sentences[sentence_num].audio_duration = audio_duration

            await self._try_start_playback(sentence_num)
            return True

    async def add_lipsync_frame(self, sentence_num: int, frame_data: str) -> bool:
        """립싱크 프레임 추가"""
        async with self.lock:
            if sentence_num not in self.sentences:
                return False

            sentence = self.sentences[sentence_num]
            sentence.frames.append(frame_data)

            if len(sentence.frames) == 1:
                logger.info(f"[SYNC] 턴#{self.current_turn} 문장#{sentence_num}: 첫 프레임 도착! 재생 시작")
                await self._try_start_playback(sentence_num)

            return True

    async def mark_lipsync_complete(self, sentence_num: int) -> bool:
        """립싱크 완료"""
        async with self.lock:
            if sentence_num not in self.sentences:
                return False

            sentence = self.sentences[sentence_num]
            sentence.is_complete = True

            # 마지막 문장 여부 동적 체크
            if self.total_sentences > 0:
                sentence.is_last_sentence = (sentence_num == self.total_sentences)

            logger.info(f"[SYNC] 턴#{self.current_turn} 문장#{sentence_num} 립싱크 완료 "
                        f"({len(sentence.frames)}프레임, 마지막={sentence.is_last_sentence}, "
                        f"전체={self.total_sentences}, LLM완료={self.llm_completed})")

            # 재생이 끝난 상태에서 마지막 문장이 완료되면 즉시 대기 상태로
            if sentence.is_last_sentence and not self.is_playing:
                logger.info(f"[SYNC] 턴#{self.current_turn}: 재생 종료 상태에서 마지막 문장 완료 감지 → 대기 전환")
                await self._enter_idle()

            return True

    async def _start_new_turn(self):
        """새 턴 시작 (완전 초기화)"""
        self.current_turn += 1

        logger.info(f"[SYNC] 턴#{self.current_turn} 시작 - 이전 턴 데이터 완전 초기화")

        # 1. 재생 상태 초기화
        self.is_playing = False
        self.current_sentence_num = None
        self.current_frame_index = 0
        self.playback_start_time = None
        self.audio_sent = False

        # 2. 타이머 정리
        if self.sync_timer and not self.sync_timer.done():
            self.sync_timer.cancel()
            try:
                await self.sync_timer
            except asyncio.CancelledError:
                pass
            self.sync_timer = None
            logger.debug(f"[SYNC] 턴#{self.current_turn}: 이전 타이머 취소")

        # 3. 문장 데이터 완전 초기화
        old_sentence_count = len(self.sentences)
        self.sentences.clear()
        logger.debug(f"[SYNC] 턴#{self.current_turn}: 이전 {old_sentence_count}개 문장 데이터 삭제")

        # 4. pending_total_sentences 처리
        if self.pending_total_sentences > 0:
            self.total_sentences = self.pending_total_sentences
            self.llm_completed = True
            self.pending_total_sentences = 0
            logger.info(f"[SYNC] 턴#{self.current_turn}: 예약된 전체 문장 수 적용 = {self.total_sentences}")
        else:
            self.total_sentences = 0
            self.llm_completed = False
            logger.debug(f"[SYNC] 턴#{self.current_turn}: 전체 문장 수 대기 중")

        # 5. idle 상태 해제는 TTS가 추가될 때 자동으로 처리됨
        logger.info(f"[SYNC] 턴#{self.current_turn}: 초기화 완료 (is_idle={self.is_idle})")

    async def _try_start_playback(self, sentence_num: int):
        """재생 시작 시도"""
        sentence = self.sentences.get(sentence_num)
        if not sentence or not sentence.tts_response:
            return

        if self.is_playing and self.current_sentence_num is not None:
            return

        if len(sentence.frames) == 0:
            logger.debug(f"[SYNC] 문장#{sentence_num}: 첫 프레임 대기 중...")
            return

        self.current_sentence_num = sentence_num
        self.current_frame_index = 0
        self.playback_start_time = time.time()
        self.is_playing = True
        self.audio_sent = False

        logger.info(f"[SYNC] 턴#{self.current_turn} 문장#{sentence_num} 재생 시작 "
                    f"(프레임 {len(sentence.frames)}개 준비됨)")

        # 1. 첫 프레임 즉시 송출
        if self.video_callback:
            try:
                first_frame = sentence.frames[0]
                if asyncio.iscoroutinefunction(self.video_callback):
                    await self.video_callback(first_frame)
                else:
                    self.video_callback(first_frame)
                logger.debug(f"[SYNC] 첫 프레임 송출 완료")
            except Exception as e:
                logger.error(f"[SYNC] 첫 프레임 송출 오류: {e}")

        # 2. 오디오 송출
        if self.audio_callback:
            try:
                if asyncio.iscoroutinefunction(self.audio_callback):
                    await self.audio_callback(sentence.tts_response.audio_data)
                else:
                    self.audio_callback(sentence.tts_response.audio_data)
                self.audio_sent = True
                logger.debug(f"[SYNC] 오디오 송출 완료")
            except Exception as e:
                logger.error(f"[SYNC] 오디오 콜백 오류: {e}")

        # 3. 동기화 타이머 시작
        if not self.sync_timer or self.sync_timer.done():
            self.sync_timer = asyncio.create_task(self._frame_loop())

    async def _frame_loop(self):
        """프레임 송출 루프 (25 FPS)"""
        try:
            while self.is_playing:
                if self.current_sentence_num is None:
                    await asyncio.sleep(self.frame_interval)
                    continue

                await self._send_frame()
                await asyncio.sleep(self.frame_interval)

        except asyncio.CancelledError:
            logger.debug(f"[SYNC] 프레임 루프 취소됨")
        except Exception as e:
            logger.error(f"[SYNC] 프레임 루프 오류: {e}")

    async def _send_frame(self):
        """프레임 송출 (실시간)"""
        if self.current_sentence_num is None or not self.video_callback:
            return

        sentence = self.sentences.get(self.current_sentence_num)
        if not sentence:
            return

        elapsed_time = time.time() - self.playback_start_time
        expected_frame_index = int(elapsed_time * self.frame_rate)

        # 프레임이 아직 도착하지 않았으면 경고
        if expected_frame_index >= len(sentence.frames):
            # 프레임 부족 경고 (1초마다)
            if not hasattr(sentence, '_last_warning_time'):
                sentence._last_warning_time = 0

            current_time = time.time()
            if current_time - sentence._last_warning_time >= 1.0:
                if sentence.is_complete:
                    logger.warning(f"[SYNC] 턴#{self.current_turn} 문장#{self.current_sentence_num}: "
                                   f"프레임 부족 - 요청: {expected_frame_index}, 실제: {len(sentence.frames)} "
                                   f"(립싱크 완료, 문장 종료 처리 중)")
                else:
                    logger.warning(f"[SYNC] 턴#{self.current_turn} 문장#{self.current_sentence_num}: "
                                   f"프레임 부족 - 요청: {expected_frame_index}, 실제: {len(sentence.frames)} "
                                   f"(립싱크 진행 중, 프레임 대기 중) ⚠️")
                sentence._last_warning_time = current_time

            if sentence.is_complete:
                await self._finish_sentence()
            return

        # 프레임 송출
        try:
            frame = sentence.frames[expected_frame_index]

            if asyncio.iscoroutinefunction(self.video_callback):
                await self.video_callback(frame)
            else:
                self.video_callback(frame)

            self.current_frame_index = expected_frame_index + 1

            # 프레임 송출 로그 (디버깅용 - 1초마다)
            if expected_frame_index % 25 == 0:
                logger.debug(f"[SYNC] 턴#{self.current_turn} 문장#{self.current_sentence_num}: "
                             f"프레임 {expected_frame_index}/{len(sentence.frames)} 재생 중")

        except IndexError as e:
            # 인덱스 에러 상세 로그
            logger.error(f"[SYNC] 프레임 인덱스 에러 - "
                         f"요청: {expected_frame_index}, 실제: {len(sentence.frames)}, "
                         f"current_index: {self.current_frame_index}")
        except Exception as e:
            logger.error(f"[SYNC] 프레임 송출 오류: {e}")

    async def _finish_sentence(self):
        """문장 재생 완료"""
        if self.current_sentence_num is None:
            return

        sentence = self.sentences.get(self.current_sentence_num)
        if not sentence:
            return

        is_last = self._is_last_sentence(self.current_sentence_num)

        logger.info(f"[SYNC] 턴#{self.current_turn} 문장#{self.current_sentence_num} 재생 완료 "
                    f"(마지막={is_last}, LLM완료={self.llm_completed}, 전체={self.total_sentences})")

        finished_sentence_num = self.current_sentence_num
        self.current_sentence_num = None
        self.current_frame_index = 0
        self.playback_start_time = None
        self.is_playing = False

        if is_last:
            await self._enter_idle()
        else:
            next_sentence_num = finished_sentence_num + 1
            next_sentence = self.sentences.get(next_sentence_num)

            if next_sentence and next_sentence.tts_response:
                await self._try_start_playback(next_sentence_num)
            else:
                logger.info(f"[SYNC] 턴#{self.current_turn}: 다음 문장 대기 중 (현재 프레임 유지)")

    def _is_last_sentence(self, sentence_num: int) -> bool:
        """마지막 문장 여부 판단"""
        if self.llm_completed and self.total_sentences > 0:
            return sentence_num >= self.total_sentences

        sentence = self.sentences.get(sentence_num)
        if sentence and sentence.is_last_sentence:
            return True

        return False

    async def _enter_idle(self):
        """대기 상태 (턴 완료)"""
        logger.info(f"[SYNC] 턴#{self.current_turn} 완료 - 대기 상태로 전환")

        # 타이머 정리
        if self.sync_timer and not self.sync_timer.done():
            self.sync_timer.cancel()
            try:
                await self.sync_timer
            except asyncio.CancelledError:
                pass
            self.sync_timer = None
            logger.debug(f"[SYNC] 턴#{self.current_turn}: 타이머 정리 완료")

        self.is_idle = True

        if self.idle_callback:
            try:
                if asyncio.iscoroutinefunction(self.idle_callback):
                    await self.idle_callback()
                else:
                    self.idle_callback()
            except Exception as e:
                logger.error(f"[SYNC] 대기 콜백 오류: {e}")

        logger.info(f"[SYNC] 턴#{self.current_turn}: 대기 상태 완료, 다음 턴 준비됨")

    def get_status(self) -> dict:
        """현재 상태"""
        return {
            'session_id': self.session_id[:4],
            'current_turn': self.current_turn,
            'is_idle': self.is_idle,
            'is_playing': self.is_playing,
            'current_sentence': self.current_sentence_num,
            'total_sentences': self.total_sentences,
            'llm_completed': self.llm_completed,
            'pending_total_sentences': self.pending_total_sentences,
            'sentences_count': len(self.sentences),
            'has_active_timer': self.sync_timer is not None and not self.sync_timer.done()
        }

    async def force_idle(self):
        """강제 대기"""
        async with self.lock:
            self.is_playing = False
            self.is_idle = True

            self.current_sentence_num = None
            self.current_frame_index = 0
            self.playback_start_time = None

            if self.sync_timer and not self.sync_timer.done():
                self.sync_timer.cancel()
                try:
                    await self.sync_timer
                except asyncio.CancelledError:
                    pass

            if self.idle_callback:
                try:
                    if asyncio.iscoroutinefunction(self.idle_callback):
                        await self.idle_callback()
                    else:
                        self.idle_callback()
                except:
                    pass

    async def cleanup(self):
        """정리"""
        logger.info(f"[SYNC] 세션 {self.session_id[:4]}: 정리")

        try:
            await self.force_idle()

            self.sentences.clear()

            self.audio_callback = None
            self.video_callback = None
            self.idle_callback = None

        except Exception as e:
            logger.error(f"[SYNC] 정리 오류: {e}")


class SyncManagerPool:
    """동기화 매니저 풀"""

    def __init__(self):
        self.managers: Dict[str, SyncManager] = {}

    def create_manager(self, session_id: str) -> SyncManager:
        if session_id in self.managers:
            return self.managers[session_id]

        manager = SyncManager(session_id)
        self.managers[session_id] = manager
        return manager

    def get_manager(self, session_id: str) -> Optional[SyncManager]:
        return self.managers.get(session_id)

    async def remove_manager(self, session_id: str):
        if session_id in self.managers:
            manager = self.managers[session_id]
            await manager.cleanup()
            del self.managers[session_id]

    async def cleanup_all(self):
        cleanup_tasks = [manager.cleanup() for manager in self.managers.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.managers.clear()


# 전역 매니저 풀
sync_manager_pool = SyncManagerPool()
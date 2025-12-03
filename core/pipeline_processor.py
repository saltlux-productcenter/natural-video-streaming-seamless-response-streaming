# pipeline_processor.py (방법 2: 중복 체크 상위 이동 버전)
import asyncio
import logging
import os
from typing import Dict, Optional, Set, Callable, List
from core.api_request.stt_processor import STTProcessor
from core.api_request.llm_processor import llm_manager
from core.api_request.tts_processor import tts_manager, TTSResponse
from core.api_request.lipsync_processor import lipsync_manager, LipsyncResponse, LipsyncFrame
from core.sync_manager import sync_manager_pool

logger = logging.getLogger(__name__)


class STTLLMTTSLipsyncPipeline:
    """STT → LLM → TTS → Lipsync 파이프라인"""

    def __init__(self, session_id: str):
        self.session_id = session_id

        # 프로세서
        self.stt_processor = STTProcessor(session_id)
        self.llm_processor = llm_manager.create_processor(session_id)
        self.tts_processor = tts_manager.create_processor(session_id)
        self.lipsync_processor = lipsync_manager.create_processor(session_id)

        # 설정
        self.min_length = int(os.getenv('MIN_LOG_LENGTH', '20'))
        self.end_punctuation = ['.', '!', '?', '。', '！', '？']

        self.tts_enabled = os.getenv('TTS_ENABLED', 'true').lower() == 'true'
        self.tts_min_length = int(os.getenv('TTS_MIN_LENGTH', '10'))

        self.lipsync_enabled = os.getenv('LIPSYNC_ENABLED', 'true').lower() == 'true'
        self.lipsync_min_length = int(os.getenv('LIPSYNC_MIN_LENGTH', '10'))

        # 문장 최대 길이 설정
        self.max_sentence_length = int(os.getenv('MAX_SENTENCE_LENGTH', '60'))

        # 문장 저장
        self.sentences = []
        self.current_streaming_text = ""
        self.last_sentence_end = 0

        # 문장 빠른 검색용 딕셔너리
        self.sentence_text_to_num: Dict[str, int] = {}

        # 스트리밍 상태
        self.streaming_in_progress = False
        self.is_llm_complete = False

        # 동시 처리 방지
        self.processing_lock = asyncio.Lock()
        self.is_processing = False

        # 문장 처리 추적
        self.sentence_to_tts = {}
        self.sentence_to_lipsync = {}

        self.tts_requested_sentences: Set[int] = set()
        self.lipsync_requested_sentences: Set[int] = set()
        self.lipsync_completed_sentences: Set[int] = set()

        self.lipsync_request_mapping: Dict[bytes, int] = {}

        # 콜백
        self.tts_complete_callbacks = []
        self.lipsync_frame_callbacks = []
        self.lipsync_complete_callbacks = []

        # 콜백 등록
        self.stt_processor.add_result_callback(self._on_stt_result)
        self.llm_processor.add_streaming_callback(self._on_llm_chunk)
        self.lipsync_processor.add_frame_callback(self._on_lipsync_frame)
        self.lipsync_processor.add_complete_callback(self._on_lipsync_complete)
        self.tts_processor.add_completion_callback(self._on_tts_completion)

        logger.info(f"[PIPELINE] 세션 {session_id[:8]}: 파이프라인 생성 (최대 문장 길이: {self.max_sentence_length}자)")

    def add_tts_complete_callback(self, callback: Callable):
        self.tts_complete_callbacks.append(callback)

    def add_lipsync_frame_callback(self, callback: Callable):
        self.lipsync_frame_callbacks.append(callback)

    def add_lipsync_complete_callback(self, callback: Callable):
        self.lipsync_complete_callbacks.append(callback)

    def _split_long_sentence(self, sentence: str, max_length: int = 60) -> List[str]:
        """
        긴 문장을 자연스러운 지점에서 단어 단위로 분할

        우선순위:
        1. 쉼표(,), 세미콜론(;) 위치
        2. 접속사 앞
        3. 단어 경계
        """
        if len(sentence) <= max_length:
            return [sentence]

        min_chunk_length = max(self.tts_min_length, 15)
        chunks = []
        natural_breaks = [',', ';', '、']
        conjunctions = [
            '그리고', '하지만', '그러나', '또한', '그런데', '따라서',
            '또는', '혹은', '게다가', '그래서', '왜냐하면'
        ]

        remaining = sentence

        while len(remaining) > max_length:
            best_split = -1

            # 1. 자연스러운 구분자 찾기
            for break_char in natural_breaks:
                search_text = remaining[:max_length]
                pos = search_text.rfind(break_char)
                if min_chunk_length <= pos <= max_length:
                    best_split = pos + 1
                    break

            # 2. 접속사 앞에서 자르기
            if best_split == -1:
                for conj in conjunctions:
                    search_text = remaining[:max_length]
                    pos = search_text.rfind(' ' + conj)
                    if min_chunk_length <= pos <= max_length:
                        best_split = pos
                        break

            # 3. 마지막 공백에서 자르기
            if best_split == -1:
                search_text = remaining[:max_length]
                pos = search_text.rfind(' ')
                if min_chunk_length <= pos <= max_length:
                    best_split = pos
                else:
                    best_split = max_length

            chunk = remaining[:best_split].strip()
            if chunk:
                chunks.append(chunk)

            remaining = remaining[best_split:].strip()

        # 마지막 남은 텍스트 처리
        if remaining:
            if len(remaining) < min_chunk_length and chunks:
                # 마지막 청크와 합치기
                last_chunk = chunks.pop()
                combined = last_chunk + ' ' + remaining

                if len(combined) > max_length * 1.2:
                    # 다시 분할
                    mid_point = len(combined) // 2
                    split_pos = combined.rfind(' ', mid_point - 10, mid_point + 10)
                    if split_pos == -1:
                        split_pos = mid_point
                    chunks.append(combined[:split_pos].strip())
                    chunks.append(combined[split_pos:].strip())
                else:
                    chunks.append(combined)
            else:
                chunks.append(remaining)

        return chunks

    async def _on_stt_result(self, text: str, is_final: bool, result_data: Dict):
        """STT 결과"""
        if not is_final or not text.strip():
            return

        logger.info(f"[PIPELINE] 세션 {self.session_id[:4]}: STT - '{text}'")

        # SyncManager 상태 확인
        sync_manager = sync_manager_pool.get_manager(self.session_id)

        # 재생 중이면 무시
        if sync_manager and not sync_manager.is_idle:
            logger.warning(f"[PIPELINE] 세션 {self.session_id[:4]}: 재생 중 - STT 무시 '{text}'")
            return

        # 이미 처리 중이면 무시
        if self.is_processing:
            logger.warning(f"[PIPELINE] 세션 {self.session_id[:4]}: 처리 중 - STT 무시 '{text}'")
            return

        # 처리 시작
        asyncio.create_task(self._process_request_queue(text))

    async def _process_request_queue(self, initial_text: str):
        """요청 큐 처리"""
        async with self.processing_lock:
            self.is_processing = True
            try:
                await self._process_single_request(initial_text)
                await self._wait_for_playback_complete()

            except Exception as e:
                logger.error(f"[PIPELINE] 요청 처리 오류: {e}")
            finally:
                self.is_processing = False
                logger.info(f"[PIPELINE] 세션 {self.session_id[:4]}: 처리 완료")

    async def _wait_for_playback_complete(self):
        """재생 완료 대기"""
        sync_manager = sync_manager_pool.get_manager(self.session_id)
        if not sync_manager:
            return

        check_interval = 0.1
        waited_time = 0

        logger.info(f"[PIPELINE] 재생 완료 대기 시작...")

        while True:
            if sync_manager.is_idle:
                logger.info(f"[PIPELINE] 재생 완료 확인 ({waited_time:.1f}초 대기)")
                return

            await asyncio.sleep(check_interval)
            waited_time += check_interval

    async def _process_single_request(self, text: str):
        """단일 요청 처리"""
        logger.info(f"[PIPELINE] 세션 {self.session_id[:4]}: 처리 시작 - '{text}'")

        # 새 대화 시작시 초기화
        if self.lipsync_enabled:
            lipsync_manager.reset_conversation(self.session_id)

        self.tts_requested_sentences.clear()
        self.lipsync_requested_sentences.clear()
        self.lipsync_completed_sentences.clear()
        self.lipsync_request_mapping.clear()

        # 상태 초기화
        self.streaming_in_progress = True
        self.is_llm_complete = False
        self.current_streaming_text = ""
        self.last_sentence_end = 0
        self.sentence_to_tts.clear()
        self.sentence_to_lipsync.clear()
        self.sentences.clear()
        self.sentence_text_to_num.clear()

        try:
            await self.llm_processor.process_text(text)

            self.streaming_in_progress = False
            self.is_llm_complete = True
            await self._process_remaining_text()

            # LLM 완료 후 전체 문장 수 전달
            total_sentences = len(self.sentences)
            sync_manager = sync_manager_pool.get_manager(self.session_id)
            if sync_manager:
                sync_manager.set_total_sentences(total_sentences)
                logger.info(f"[PIPELINE] 세션 {self.session_id[:4]}: 전체 {total_sentences}개 문장")

        except Exception as e:
            logger.error(f"[PIPELINE] 처리 실패: {e}")
            self.streaming_in_progress = False
            self.is_llm_complete = True

    async def _on_llm_chunk(self, chunk_content: str, streaming_text: str, chunk_count: int):
        """LLM 청크"""
        if not self.streaming_in_progress:
            return

        self.current_streaming_text = streaming_text
        await self._check_for_new_sentences()

    async def _check_for_new_sentences(self):
        """새 문장 감지 (✅ 중복 체크 포함)"""
        text = self.current_streaming_text
        search_start = self.last_sentence_end

        for punct in self.end_punctuation:
            pos = search_start
            while True:
                pos = text.find(punct, pos)
                if pos == -1:
                    break

                sentence = text[self.last_sentence_end:pos + 1].strip()

                if len(sentence) >= self.min_length:
                    # ✅ 중복 체크를 여기서 먼저!
                    clean_sentence = sentence.replace(" [미완성]", "").strip()

                    # 이미 저장된 문장이면 건너뛰기
                    if clean_sentence in self.sentences:
                        logger.debug(f"[PIPELINE] 중복 문장 건너뜀: '{clean_sentence[:30]}...'")
                        self.last_sentence_end = pos + 1
                        pos += 1
                        continue

                    # 중복이 아니면 저장
                    await self._save_and_log_sentence(sentence)
                    self.last_sentence_end = pos + 1

                pos += 1

    async def _save_and_log_sentence(self, sentence: str):
        """문장 저장 및 TTS 처리 (✅ 중복 체크 없음 - 상위에서 이미 체크함)"""
        clean_sentence = sentence.replace(" [미완성]", "").strip()

        if len(clean_sentence) > self.max_sentence_length:
            logger.info(f"[PIPELINE] 긴 문장 감지 ({len(clean_sentence)}자) - 분할 처리: {clean_sentence[:50]}...")

            sub_sentences = self._split_long_sentence(clean_sentence, self.max_sentence_length)

            for i, sub_sentence in enumerate(sub_sentences):
                self.sentences.append(sub_sentence)
                sentence_num = len(self.sentences)

                normalized = ' '.join(sub_sentence.split())
                self.sentence_text_to_num[normalized] = sentence_num

                display_sentence = f"{sub_sentence} [{i + 1}/{len(sub_sentences)}]"
                logger.info(f"[PIPELINE] 문장 #{sentence_num}: {display_sentence}")

                # TTS 처리
                if (self.tts_enabled and
                        len(sub_sentence) >= self.tts_min_length and
                        sentence_num not in self.tts_requested_sentences):
                    self.tts_requested_sentences.add(sentence_num)
                    await self.tts_processor.process_text_to_speech(sub_sentence)
        else:
            self.sentences.append(clean_sentence)
            sentence_num = len(self.sentences)

            normalized = ' '.join(clean_sentence.split())
            self.sentence_text_to_num[normalized] = sentence_num

            logger.info(f"[PIPELINE] 문장 #{sentence_num}: {sentence}")

            # TTS 처리
            if (self.tts_enabled and
                    len(clean_sentence) >= self.tts_min_length and
                    sentence_num not in self.tts_requested_sentences):
                self.tts_requested_sentences.add(sentence_num)
                await self.tts_processor.process_text_to_speech(clean_sentence)

    async def _on_tts_completion(self, tts_response: TTSResponse):
        """TTS 완료 즉시 처리"""
        try:
            # O(1) 해시 테이블 검색
            normalized_text = ' '.join(tts_response.original_text.split())
            sentence_num = self.sentence_text_to_num.get(normalized_text)

            if sentence_num is None:
                logger.warning(f"[PIPELINE] TTS 완료됐지만 매칭되는 문장을 찾을 수 없음: '{normalized_text[:50]}...'")
                return

            # 중복 체크
            if sentence_num in self.sentence_to_tts:
                return

            # 원자적 할당
            self.sentence_to_tts[sentence_num] = tts_response
            tts_response.sentence_num = sentence_num

            logger.info(f"[PIPELINE] 문장 #{sentence_num} TTS 완료 ⚡ (즉시 감지)")

            # TTS 완료 콜백 호출
            for callback in self.tts_complete_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(sentence_num, tts_response)
                    else:
                        callback(sentence_num, tts_response)
                except Exception as e:
                    logger.error(f"[PIPELINE] TTS 콜백 오류: {e}")

            # Lipsync 처리
            if (self.lipsync_enabled and
                    len(tts_response.original_text) >= self.lipsync_min_length and
                    sentence_num not in self.lipsync_requested_sentences and
                    sentence_num not in self.lipsync_completed_sentences):
                self.lipsync_requested_sentences.add(sentence_num)

                asyncio.create_task(self._process_lipsync_for_sentence(
                    tts_response.audio_data,
                    sentence_num,
                    tts_response.original_text,
                    tts_response.affective_state,
                ))

        except Exception as e:
            logger.error(f"[PIPELINE] TTS 완료 처리 오류: {e}")

    async def _process_lipsync_for_sentence(self,
        audio_data: bytes,
        sentence_num: int,
        text: str,
        affective_state: str = "neutral",
    ):
        """Lipsync 처리 (문장 번호 전달)"""
        try:
            if sentence_num in self.lipsync_completed_sentences:
                return

            # 매핑 등록 (fallback용)
            self.lipsync_request_mapping[audio_data] = sentence_num

            is_last_sentence = (
                    self.is_llm_complete and
                    sentence_num == len(self.sentences)
            )

            # 문장 번호를 Lipsync 프로세서에 전달
            await self.lipsync_processor.process_audio_to_lipsync(
                audio_data,
                is_last_sentence,
                sentence_num,
                affective_state
            )

        except Exception as e:
            logger.error(f"[PIPELINE] Lipsync 오류: {e}")

    async def _on_lipsync_frame(self, frame: LipsyncFrame, session_id: str):
        """Lipsync 프레임 (frame.sentence_num 사용)"""
        if session_id != self.session_id:
            return

        # 프레임에 포함된 문장 번호 직접 사용
        sentence_num = frame.sentence_num

        if sentence_num is None:
            logger.warning(f"[PIPELINE] 문장 번호 없는 프레임 무시 (프레임#{frame.index})")
            return

        # 프레임 콜백
        for callback in self.lipsync_frame_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(sentence_num, frame.data)
                else:
                    callback(sentence_num, frame.data)
            except Exception as e:
                logger.error(f"[PIPELINE] 프레임 콜백 오류: {e}")

    async def _on_lipsync_complete(self, lipsync_response: LipsyncResponse):
        """Lipsync 완료"""
        if lipsync_response.session_id != self.session_id:
            return

        # LipsyncResponse의 sentence_num 직접 사용
        sentence_num = lipsync_response.sentence_num

        # Fallback: 오디오 데이터로 문장 찾기
        if sentence_num is None:
            sentence_num = self.lipsync_request_mapping.get(lipsync_response.original_audio_data)

        if sentence_num is None:
            logger.warning(f"[PIPELINE] 문장 번호를 알 수 없는 Lipsync 완료")
            return

        if sentence_num in self.lipsync_completed_sentences:
            return

        # 매핑 제거
        if lipsync_response.original_audio_data in self.lipsync_request_mapping:
            del self.lipsync_request_mapping[lipsync_response.original_audio_data]

        self.sentence_to_lipsync[sentence_num] = lipsync_response
        self.lipsync_completed_sentences.add(sentence_num)

        logger.info(f"[PIPELINE] 문장 #{sentence_num} Lipsync 완료 ({lipsync_response.total_frames}프레임)")

        # Lipsync 완료 콜백
        for callback in self.lipsync_complete_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(sentence_num)
                else:
                    callback(sentence_num)
            except Exception as e:
                logger.error(f"[PIPELINE] Lipsync 콜백 오류: {e}")

    async def _process_remaining_text(self):
        """남은 텍스트 처리 (✅ 중복 체크 포함)"""
        if not self.current_streaming_text or self.last_sentence_end >= len(self.current_streaming_text):
            return

        remaining = self.current_streaming_text[self.last_sentence_end:].strip()

        if len(remaining) >= self.min_length:
            # ✅ 중복 체크
            if remaining not in self.sentences:
                await self._save_and_log_sentence(remaining + " [미완성]")

    def get_sentence_count(self) -> int:
        """문장 수"""
        return len(self.sentences)

    def get_pipeline_stats(self) -> Dict:
        """상태 정보"""
        sync_manager = sync_manager_pool.get_manager(self.session_id)

        return {
            "session_id": self.session_id[:4],
            "sentence_count": len(self.sentences),
            "is_processing": self.is_processing,
            "is_idle": sync_manager.is_idle if sync_manager else True
        }

    async def start_stt(self) -> bool:
        """STT 시작"""
        return await self.stt_processor.connect_stt()

    async def cleanup(self):
        """정리"""
        logger.info(f"[PIPELINE] 세션 {self.session_id[:4]}: 정리")

        try:
            wait_count = 0
            while self.is_processing and wait_count < 100:
                await asyncio.sleep(0.1)
                wait_count += 1

            await self._process_remaining_text()

            # 콜백 제거
            self.stt_processor.remove_result_callback(self._on_stt_result)
            self.llm_processor.remove_streaming_callback(self._on_llm_chunk)
            self.lipsync_processor.remove_frame_callback(self._on_lipsync_frame)
            self.lipsync_processor.remove_complete_callback(self._on_lipsync_complete)
            self.tts_processor.remove_completion_callback(self._on_tts_completion)

            # 콜백 리스트 정리
            self.tts_complete_callbacks.clear()
            self.lipsync_frame_callbacks.clear()
            self.lipsync_complete_callbacks.clear()
            self.sentence_to_tts.clear()
            self.sentence_to_lipsync.clear()

            # 추적 세트 정리
            self.tts_requested_sentences.clear()
            self.lipsync_requested_sentences.clear()
            self.lipsync_completed_sentences.clear()
            self.lipsync_request_mapping.clear()
            self.sentence_text_to_num.clear()

            # 프로세서 정리
            await self.stt_processor.close()
            await llm_manager.remove_processor(self.session_id)
            await tts_manager.remove_processor(self.session_id)
            await lipsync_manager.remove_processor(self.session_id)

        except Exception as e:
            logger.error(f"[PIPELINE] 정리 오류: {e}")


class PipelineManager:
    """파이프라인 매니저"""

    def __init__(self):
        self.pipelines: Dict[str, STTLLMTTSLipsyncPipeline] = {}

    def create_pipeline(self, session_id: str) -> STTLLMTTSLipsyncPipeline:
        if session_id in self.pipelines:
            return self.pipelines[session_id]

        pipeline = STTLLMTTSLipsyncPipeline(session_id)
        self.pipelines[session_id] = pipeline
        return pipeline

    def get_pipeline(self, session_id: str) -> Optional[STTLLMTTSLipsyncPipeline]:
        return self.pipelines.get(session_id)

    async def remove_pipeline(self, session_id: str):
        if session_id in self.pipelines:
            await self.pipelines[session_id].cleanup()
            del self.pipelines[session_id]

    async def cleanup_all(self):
        cleanup_tasks = [pipeline.cleanup() for pipeline in self.pipelines.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.pipelines.clear()


# 전역 파이프라인 매니저
pipeline_manager = PipelineManager()
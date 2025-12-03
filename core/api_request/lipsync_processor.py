# lipsync_processor.py (오디오 데이터 기반 매핑 버전)
import asyncio
import json
import logging
import time
import aiohttp
import os
import io
import random
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LipsyncFrame:
    """립싱크 프레임 데이터"""
    index: int
    data: str  # base64 encoded image
    format: str = "jpg"
    sentence_num: Optional[int] = None  # ✅ 문장 번호 추가


@dataclass
class LipsyncResponse:
    """립싱크 응답 데이터 클래스"""
    original_audio_data: bytes
    cache_key: str
    frames: List[LipsyncFrame] = field(default_factory=list)
    timestamp: float = 0.0
    session_id: str = ""
    processing_time: float = 0.0
    start_frame_index: int = 0
    last_frame_index: int = 0
    next_start_frame: int = 0
    total_frames: int = 0
    process_full_video: bool = False
    avatar_end_complete: bool = False

    # 문장 번호 필드 (파이프라인에서 설정)
    sentence_num: Optional[int] = None


class LipsyncProcessor:
    """Lipsync 서버 처리만 담당하는 클래스"""

    def __init__(self, session_id: str, base_url: Optional[str] = None):
        self.session_id = session_id
        self.base_url = os.getenv('LIPSYNC_BASE_URL')
        if self.base_url == "None":
            raise ValueError("LIPSYNC_BASE_URL 환경 변수가 설정되지 않았습니다.")
        self.endpoint = f"{self.base_url}/video/generate"

        # 레퍼런스 비디오 (캐시키)
        # TODO: 외부 파일에서 설정을 읽어오도록 수정
        # self.reference_video = "9701029eb4c30e464cde54b6d5abee370c32e5674772854090abcd99adc97c22"
        self.reference_videos = dict(
            neutral=[
                "1a7ef61ffda377835319d4d7bbf8776ca9cc53f1d76039c767180094ec738102",
                "187f66845fecaaff4661200d329982623d452f449c4fa26d62a8ed543347eca5",
                "99fed1d9af82554747cff50d3042b5e0a6996b27e6b2df2d5dab0857f3ad638a",
                "63d925073645c9808ca9887c4a765374c72ec5c7bb2fdd04af611c41df6f8de2",
            ],
            happiness=[
                "588a266caab183b64db0cd7cf946c61ca6024c75ceddefc7864268695ba28ae6",
                "89597daf79291d28337ec9009f2bd74db15f176e92ba2b061ad27f71fd0b2b97",
                "6b1b03ab6380725e8ce3092ea1ef5ffbb7d4142b75c84ea97b9a6cc384147ea1",
                "9f52839cc6bcf9cbea8da3745fcfd89bbacf1962751cc01490bdad9c9b51ec41",
            ],
            sadness=[
                "4dde9b480c9f919baf833153fc67a6ed12a7539fcbfcd0be33fe2919f5bb1101",
            ],
            anger=[
                "63ac932c0ac6547869757f037deaf11dc6d3c6472643b4631e637e9d5166f586",
                "b5caddeb8c52e0930f077eb7f7def1df085f46c6bdee29d7c9325e6099cdc963",
                "6e762bbe2150867fe17b9892ccefc765c7dfdb5074313f1fd4e381b893b7a77e",
                "71f0e9209ae4100b1f67c31a30434f00d9c563ce9c1b13ac9005582a5e308281",
            ],
            surprise=[
                "d4c1ad1634593acdfa8ea5c4ff1cd86e74bba277de75da5e7b4a0a3a53b9666d",
            ],
            afraid=[
                "3f2fee323cbfbc56ae6dfd8a5739b65ac4dac7615db9b8f027be28adda8d2804",
            ],
            fear=[
                "3f2fee323cbfbc56ae6dfd8a5739b65ac4dac7615db9b8f027be28adda8d2804",
            ],
            disgust=[
                "57fb8a5af2ec8721cb239eb50b480df5ae914e6fcb69d625a91359b82096f8e4",
            ],
            contempt=[
                "a32c131e8807d0df6b31737b0a26909c43c516bcffe096cf45e3bcc2e168e8c2",
            ],
            shame=[
                "4dde9b480c9f919baf833153fc67a6ed12a7539fcbfcd0be33fe2919f5bb1101",
            ],
            hope=[
                "6b1b03ab6380725e8ce3092ea1ef5ffbb7d4142b75c84ea97b9a6cc384147ea1",
            ],
            interest=[
                "9f52839cc6bcf9cbea8da3745fcfd89bbacf1962751cc01490bdad9c9b51ec41",
            ],
            boredom=[
                "1b47f502d0c0b93c3c680117279111227776e0053ddaaf33e24a50b55e48403a",
            ],
            thinking=[
                "258ed19031ed38bba3cded11061107fa8b1768d8e2c4289269274a52276756ff",
            ],
            joking=[
                "89597daf79291d28337ec9009f2bd74db15f176e92ba2b061ad27f71fd0b2b97",
            ],
            encourage=[
                "6b1b03ab6380725e8ce3092ea1ef5ffbb7d4142b75c84ea97b9a6cc384147ea1",
            ],
        )
        self.reference_videos["happy"] = self.reference_videos["happiness"]
        self.reference_videos["glad"] = self.reference_videos["happiness"]
        self.reference_videos["sad"] = self.reference_videos["sadness"]
        self.reference_videos["angry"] = self.reference_videos["anger"]
        self.reference_videos["surprised"] = self.reference_videos["surprise"]
        self.reference_videos["disgusted"] = self.reference_videos["disgust"]
        self.reference_videos['horrified'] = self.reference_videos['fear']
        self.reference_videos['interested'] = self.reference_videos['interest']
        self.reference_videos['bored'] = self.reference_videos['boredom']
        self.reference_videos['thoughtful'] = self.reference_videos['thinking']


        # 프레임 인덱스 관리
        self.current_start_frame_index = 0
        self.is_new_conversation = True

        # 동시 요청 제한을 위한 세마포어
        self.request_semaphore = asyncio.Semaphore(1)

        # 처리된 응답들
        self.processed_responses: List[LipsyncResponse] = []

        # 요청 큐와 상태 관리
        self.pending_requests = []  # (audio_data, is_last_sentence, sentence_num) 튜플들
        self.is_processing = False
        self.max_pending_requests = int(os.getenv('MAX_LIPSYNC_PENDING', '1000'))

        # ✅ 오디오 데이터 → 문장 번호 매핑
        self.audio_to_sentence: Dict[bytes, int] = {}

        # ✅ 현재 처리 중인 문장 번호 (프레임 생성 시 사용)
        self.current_sentence_num: Optional[int] = None

        # aiohttp 세션
        self.http_session: Optional[aiohttp.ClientSession] = None

        # 콜백 시스템
        self.frame_callbacks: List[Callable] = []
        self.complete_callbacks: List[Callable] = []

        logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: Lipsync 프로세서 초기화 완료")

    def reset_conversation(self):
        """새로운 대화 시작"""
        self.current_start_frame_index = 0
        self.is_new_conversation = True
        self.audio_to_sentence.clear()  # ✅ 매핑도 초기화
        logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 새로운 대화 시작 - 프레임 인덱스 리셋")

    def add_frame_callback(self, callback: Callable):
        """프레임별 콜백 함수 추가"""
        self.frame_callbacks.append(callback)

    def add_complete_callback(self, callback: Callable):
        """완료 콜백 함수 추가"""
        self.complete_callbacks.append(callback)

    def remove_frame_callback(self, callback: Callable):
        """프레임별 콜백 함수 제거"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)

    def remove_complete_callback(self, callback: Callable):
        """완료 콜백 함수 제거"""
        if callback in self.complete_callbacks:
            self.complete_callbacks.remove(callback)

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 가져오기"""
        if self.http_session is None or self.http_session.closed:
            timeout = aiohttp.ClientTimeout(total=60.0)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
        return self.http_session

    async def process_audio_to_lipsync(
        self,
        audio_data: bytes,
        is_last_sentence: bool = False,
        sentence_num: Optional[int] = None,  # ✅ 문장 번호 매개변수 추가
        affective_state: str = "neutral",
    ) -> Optional[LipsyncResponse]:
        """오디오 데이터를 립싱크로 처리"""
        if not audio_data:
            return None

        # ✅ 문장 번호 매핑 저장
        if sentence_num is not None:
            self.audio_to_sentence[audio_data] = sentence_num
            logger.debug(f"[LIPSYNC] 세션 {self.session_id[:4]}: 오디오→문장 매핑 등록 (문장#{sentence_num})")

        if self.is_processing:
            if len(self.pending_requests) >= self.max_pending_requests:
                removed_request = self.pending_requests.pop(0)
                logger.warning(f"[LIPSYNC] 세션 {self.session_id[:4]}: 대기 큐 가득함, 오래된 요청 제거")

            self.pending_requests.append((audio_data, is_last_sentence, sentence_num, affective_state))  # ✅ 튜플에 sentence_num 추가
            logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 처리 중이므로 요청 대기 (대기 수: {len(self.pending_requests)})")
            return None

        asyncio.create_task(self._process_request_queue((audio_data, is_last_sentence, sentence_num, affective_state)))
        return None

    async def _process_request_queue(self, initial_request):
        """요청 큐를 순차적으로 처리"""
        async with self.request_semaphore:
            self.is_processing = True

            try:
                # 첫 번째 요청 처리
                audio_data, is_last_sentence, sentence_num, affective_state = initial_request
                logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 초기 요청 처리 시작 (문장#{sentence_num})")
                await self._process_single_request(audio_data, is_last_sentence, sentence_num, affective_state)
                logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 초기 요청 처리 완료, 대기 큐: {len(self.pending_requests)}개")

                # 대기 중인 요청들 순차 처리
                processed_count = 0
                while self.pending_requests:
                    next_audio_data, next_is_last, next_sentence_num, next_affective_state = self.pending_requests.pop(0)
                    processed_count += 1
                    logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: "
                                f"대기 요청 #{processed_count} 처리 시작 (문장#{next_sentence_num}, 남은 대기: {len(self.pending_requests)})")

                    await self._process_single_request(next_audio_data, next_is_last, next_sentence_num, next_affective_state)
                    logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: "
                                f"대기 요청 #{processed_count} 처리 완료")

                    await asyncio.sleep(0.1)

                logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 모든 대기 요청 처리 완료 (총 {processed_count}개)")

            except Exception as e:
                logger.error(f"[LIPSYNC] 세션 {self.session_id[:4]}: 요청 큐 처리 중 오류 - {e}", exc_info=True)
            finally:
                remaining = len(self.pending_requests)
                self.is_processing = False

                if remaining > 0:
                    logger.error(f"⚠️ [LIPSYNC] 세션 {self.session_id[:4]}: "
                                 f"큐 처리 완료했으나 {remaining}개 요청이 남아있음!")
                else:
                    logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 요청 큐 처리 완료 (남은 요청: 0개)")

    async def _process_single_request(
        self,
        audio_data: bytes,
        is_last_sentence: bool,
        sentence_num: Optional[int] = None,  # ✅ 문장 번호 매개변수 추가,
        affective_state: str = "neutral",
    ) -> Optional[LipsyncResponse]:
        """단일 립싱크 요청 처리"""
        try:
            start_time = time.time()
            start_frame_index = self.current_start_frame_index

            # ✅ 문장 번호 설정 (매핑에서 조회 또는 매개변수 사용)
            if sentence_num is None:
                sentence_num = self.audio_to_sentence.get(audio_data)

            self.current_sentence_num = sentence_num

            logger.info(f"🎭 [LIPSYNC] 세션 {self.session_id[:4]}: 립싱크 서버 요청 시작 "
                        f"(문장#{sentence_num}, 오디오: {len(audio_data)} bytes, "
                        f"시작프레임: {start_frame_index}, 마지막문장: {is_last_sentence}, "
                        f"감정상태: {affective_state})")

            # HTTP 세션 준비
            session = await self._get_http_session()

            # 멀티파트 데이터 준비
            data = aiohttp.FormData()

            if affective_state not in self.reference_videos:
                logger.warning(f"[LIPSYNC] 세션 {self.session_id[:4]}: 등록되지 않은 감정상태 - {affective_state}")
                affective_state = "neutral"
            ref_video_key = random.choice(self.reference_videos[affective_state])

            data.add_field('reference_video_key', ref_video_key)
            data.add_field('audio', io.BytesIO(audio_data), filename='audio.wav', content_type='audio/wav')

            params = {
                "start_frame_index": start_frame_index,
                "process_full_video": is_last_sentence
            }
            data.add_field('params', json.dumps(params), content_type='application/json')

            # 립싱크 서버에 POST 요청
            async with session.post(self.endpoint, data=data) as response:
                if response.status == 200:
                    lipsync_response = LipsyncResponse(
                        original_audio_data=audio_data,
                        cache_key=ref_video_key,
                        timestamp=time.time(),
                        session_id=self.session_id,
                        start_frame_index=start_frame_index,
                        process_full_video=is_last_sentence,
                        sentence_num=sentence_num  # ✅ 문장 번호 설정
                    )

                    # 스트리밍 응답 처리
                    await self._process_streaming_response(response, lipsync_response)

                    # 처리 완료
                    lipsync_response.processing_time = time.time() - start_time
                    self.processed_responses.append(lipsync_response)

                    # 다음 요청을 위한 프레임 인덱스 업데이트
                    if lipsync_response.next_start_frame > 0:
                        self.current_start_frame_index = lipsync_response.next_start_frame

                    if self.is_new_conversation:
                        self.is_new_conversation = False

                    logger.info(f"✅ [LIPSYNC-{self.session_id[:4]}] 문장#{sentence_num} 립싱크 완료 "
                                f"({lipsync_response.total_frames}프레임, {lipsync_response.processing_time:.2f}초, "
                                f"다음시작: {lipsync_response.next_start_frame})")

                    return lipsync_response

                else:
                    error_text = await response.text()
                    logger.error(f"[LIPSYNC] 세션 {self.session_id[:4]}: 서버 오류 {response.status} - {error_text}")
                    return None

        except asyncio.TimeoutError:
            logger.error(f"[LIPSYNC] 세션 {self.session_id[:4]}: 서버 요청 타임아웃")
            return None

        except aiohttp.ClientError as e:
            logger.error(f"[LIPSYNC] 세션 {self.session_id[:4]}: 서버 연결 오류 - {e}")
            return None

        except Exception as e:
            logger.error(f"[LIPSYNC] 세션 {self.session_id[:4]}: 처리 오류 - {e}", exc_info=True)
            return None
        finally:
            # ✅ 처리 완료 후 현재 문장 번호 초기화
            self.current_sentence_num = None

    async def _process_streaming_response(self, response: aiohttp.ClientResponse, lipsync_response: LipsyncResponse):
        """스트리밍 응답 처리"""
        frame_count = 0
        is_complete = False
        last_frame_index = 0
        next_start_frame = 0
        avatar_end_complete = False

        async for line in response.content:
            line = line.decode('utf-8').strip()
            if not line or not line.startswith('data: '):
                continue

            try:
                json_data = json.loads(line[6:])
                data_type = json_data.get('type')

                if data_type == 'connected':
                    logger.debug(f"[LIPSYNC] 세션 {self.session_id[:4]}: 연결됨")

                elif data_type == 'cache_loaded':
                    logger.debug(f"[LIPSYNC] 세션 {self.session_id[:4]}: 캐시 로드됨")

                elif data_type == 'metadata':
                    lipsync_response.total_frames = json_data.get('total_frames', 0)
                    logger.debug(
                        f"[LIPSYNC] 세션 {self.session_id[:4]}: 메타데이터 수신 (총 프레임: {lipsync_response.total_frames})")

                elif data_type == 'frame':
                    # ✅ 프레임 데이터 처리 (문장 번호 포함)
                    frame_index = json_data.get('index', 0)
                    frame_data = json_data.get('data', '')
                    frame_format = json_data.get('format', 'jpg')

                    frame = LipsyncFrame(
                        index=frame_index,
                        data=frame_data,
                        format=frame_format,
                        sentence_num=self.current_sentence_num  # ✅ 현재 처리 중인 문장 번호 설정
                    )
                    lipsync_response.frames.append(frame)
                    frame_count += 1

                    # 프레임 콜백 호출
                    for callback in self.frame_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(frame, lipsync_response.session_id)
                            else:
                                callback(frame, lipsync_response.session_id)
                        except Exception as e:
                            logger.error(f"[LIPSYNC] 프레임 콜백 오류: {e}")

                elif data_type == 'complete':
                    metadata = json_data.get('metadata', {})
                    last_frame_index = metadata.get('last_frame_index', 0)
                    next_start_frame = metadata.get('next_start_frame', 0)
                    avatar_end_complete = metadata.get('avatar_end_complete', False)
                    is_complete = True

                    logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 스트리밍 완료 "
                                f"(문장#{self.current_sentence_num}, 받은프레임: {frame_count}개, "
                                f"마지막프레임: {last_frame_index}, 다음시작: {next_start_frame})")

                elif data_type == 'final_memory_status':
                    logger.debug(f"[LIPSYNC] 세션 {self.session_id[:4]}: 최종 메모리 상태")

            except json.JSONDecodeError as e:
                logger.warning(f"[LIPSYNC] 세션 {self.session_id[:4]}: JSON 파싱 오류 - {line[:100]}...")
            except Exception as e:
                logger.error(f"[LIPSYNC] 세션 {self.session_id[:4]}: 스트리밍 데이터 처리 오류 - {e}")

        # 스트리밍 완료 후 메타데이터 설정 및 콜백 호출
        if is_complete:
            lipsync_response.last_frame_index = last_frame_index
            lipsync_response.next_start_frame = next_start_frame
            lipsync_response.avatar_end_complete = avatar_end_complete

            for callback in self.complete_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(lipsync_response)
                    else:
                        callback(lipsync_response)
                except Exception as e:
                    logger.error(f"[LIPSYNC] 완료 콜백 오류: {e}")

    async def cleanup(self):
        """립싱크 프로세서 정리"""
        logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: 정리 시작")

        try:
            wait_count = 0
            while self.is_processing and wait_count < 300:
                await asyncio.sleep(0.1)
                wait_count += 1

            if self.is_processing:
                logger.warning(f"[LIPSYNC] 세션 {self.session_id[:4]}: 처리 완료 대기 시간 초과")

            if self.pending_requests:
                logger.info(f"[LIPSYNC] 세션 {self.session_id[:4]}: {len(self.pending_requests)}개의 대기 요청 정리")
                self.pending_requests.clear()

            # ✅ 매핑 정리
            self.audio_to_sentence.clear()

            self.frame_callbacks.clear()
            self.complete_callbacks.clear()

            if self.http_session and not self.http_session.closed:
                await self.http_session.close()

        except Exception as e:
            logger.error(f"[LIPSYNC] 세션 {self.session_id[:4]}: 정리 중 오류 - {e}")


class LipsyncManager:
    """여러 세션의 립싱크 프로세서 관리"""

    def __init__(self):
        self.processors: Dict[str, LipsyncProcessor] = {}

    def create_processor(self, session_id: str) -> LipsyncProcessor:
        """새 립싱크 프로세서 생성"""
        if session_id in self.processors:
            logger.warning(f"[LIPSYNC] 프로세서가 이미 존재: {session_id[:8]}")
            return self.processors[session_id]

        processor = LipsyncProcessor(session_id)
        self.processors[session_id] = processor
        logger.info(f"[LIPSYNC] 새 프로세서 생성: {session_id[:8]}")
        return processor

    def get_processor(self, session_id: str) -> Optional[LipsyncProcessor]:
        """립싱크 프로세서 가져오기"""
        return self.processors.get(session_id)

    async def process_audio_to_lipsync(
        self,
        session_id: str,
        audio_data: bytes,
        is_last_sentence: bool = False,
        sentence_num: Optional[int] = None,  # ✅ 문장 번호 추가,
        affective_state: str = "neutral",
    ) -> Optional[LipsyncResponse]:
        """특정 세션에서 오디오를 립싱크로 변환"""
        processor = self.get_processor(session_id)
        if not processor:
            logger.error(f"[LIPSYNC] 존재하지 않는 세션: {session_id[:8]}")
            return None

        return await processor.process_audio_to_lipsync(audio_data, is_last_sentence, sentence_num, affective_state)

    def reset_conversation(self, session_id: str):
        """특정 세션의 대화 리셋"""
        processor = self.get_processor(session_id)
        if processor:
            processor.reset_conversation()

    async def remove_processor(self, session_id: str):
        """립싱크 프로세서 제거"""
        if session_id in self.processors:
            await self.processors[session_id].cleanup()
            del self.processors[session_id]
            logger.info(f"[LIPSYNC] 프로세서 제거: {session_id[:8]}")

    async def cleanup_all(self):
        """모든 립싱크 프로세서 정리"""
        cleanup_tasks = [processor.cleanup() for processor in self.processors.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.processors.clear()
        logger.info("[LIPSYNC] 모든 프로세서 정리 완료")


# 전역 립싱크 매니저
lipsync_manager = LipsyncManager()
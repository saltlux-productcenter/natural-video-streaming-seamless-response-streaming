# tts_processor.py
import asyncio
import json
import logging
import time
import aiohttp

import openai
from openai import AsyncOpenAI

import os
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TTSResponse:
    """TTS 응답 데이터 클래스"""
    original_text: str
    audio_data: bytes
    timestamp: float
    session_id: str
    processing_time: float
    voice_id: int
    sample_rate: str
    audio_format: str
    affective_state: str = "neutral"


class TTSProcessor:
    """TTS 서버 처리만 담당하는 클래스"""

    def __init__(self, session_id: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.session_id = session_id
        self.base_url = os.getenv('TTS_BASE_URL')
        if self.base_url is None:
            raise ValueError("TTS_BASE_URL 환경 변수가 설정되지 않았습니다. ")
        self.endpoint = f"{self.base_url}/v1/audio/speech"
        
        # 감정 검출용 LLM 설정
        self.llm_base_url = os.getenv('LLM_BASE_URL')
        if self.llm_base_url is None:
            raise ValueError("LLM_BASE_URL 환경 변수가 설정되지 않았습니다. ")
        self.llm_api_key = api_key or os.getenv('LLM_API_KEY', 'dummy-key')
        self.llm_client = AsyncOpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )
        self.llm_model = "LUXIA"
#         self.system_prompt = """
# You are a Human-Machine Interaction Translator. Your mission is to classify the given text into one of the following \"Affective States\".
# [ neutral, happiness, sadness, anger, surprise, afraid, fear, disgust, contempt, shame, hope, interest, boredom, thinking, joking, encourage ]
# The output *MUST* be a single word, and *MUST* be from the list.
#         """.strip()
        self.system_prompt = """당신은 주어진 [질문]에 대한 감정을 분류하는 감정사입니다. [규칙]을 준수하여 답해주세요.

<규칙>
1. [질문]에 부합하는 감정을 분류합니다.
2. 감정은 다음 [리스트]에 포함되어 있는 감정만 선택해야 합니다.
  [리스트]
  - neutral
  - happiness
  - sadness
  - anger
  - surprise
  - afraid
  - fear
  - disgust
  - contempt
  - shame
  - hope
  - boredom
  - thinking
  - joking
  - encourage
3. 감정 하나만 분류하며, 단일성 단어만 답변해야 합니다. 문장으로 답변하면 안됩니다.
4. 대학 교수의 입장에서 감정을 분류해야 합니다.
5. 감정이 불확실하거나, 객관적인 사실을 말할 경우, "neutral" 로 분류해야 합니다.
"""
        self.valid_affective_states = {
            "neutral",
            "happiness", "happy", "glad",
            "sadness", "sadness",
            "anger", "angry",
            "surprise", "surprised",
            "afraid",
            "fear", "horrified",
            "disgust", "disgusted",
            "contempt",
            "shame", "ashamed",
            "hope", "hopeful",
            "interest", "interested", "intrigued",
            "boredom", "bored",
            "thinking", "thoughtful",
            "joking", "joke",
            "encourage",
        }

        # TTS 설정
        self.voice = int(os.getenv('TTS_VOICE', '0'))
        self.pad_silence = float(os.getenv('TTS_PAD_SILENCE', '0.2'))
        self.tempo = float(os.getenv('TTS_TEMPO', '1.0'))
        self.gain_db = float(os.getenv('TTS_GAIN_DB', '0'))
        self.sample_rate = os.getenv('TTS_SAMPLE_RATE', '24k')
        self.return_type = os.getenv('TTS_RETURN_TYPE', 'wav')
        self.response_format = os.getenv('TTS_RESPONSE_FORMAT', 'wav')
        self.lang = os.getenv('TTS_LANG', 'ko')
        self.cache = os.getenv('TTS_CACHE', 'false').lower() == 'true'
        self.stream = os.getenv('TTS_STREAM', 'false').lower() == 'true'

        # 동시 요청 제한을 위한 세마포어 (한 세션당 1개)
        self.request_semaphore = asyncio.Semaphore(1)

        # 처리된 응답들
        self.processed_responses: List[TTSResponse] = []

        # ✅ 완료 콜백 추가
        self.completion_callbacks: List[Callable] = []

        # 요청 큐와 상태 관리
        self.pending_requests = []  # 대기 중인 요청들
        self.is_processing = False
        self.max_pending_requests = int(os.getenv('MAX_TTS_PENDING', '1000'))

        # aiohttp 세션
        self.http_session: Optional[aiohttp.ClientSession] = None

        logger.info(f"[TTS] 세션 {self.session_id[:4]}: TTS 프로세서 초기화 완료 "
                    f"(음성: {self.voice}, 샘플레이트: {self.sample_rate})")

    # ✅ 콜백 관리 메서드 추가
    def add_completion_callback(self, callback: Callable):
        """TTS 완료 콜백 등록"""
        if callback not in self.completion_callbacks:
            self.completion_callbacks.append(callback)
            logger.debug(f"[TTS] 세션 {self.session_id[:4]}: 완료 콜백 등록")

    def remove_completion_callback(self, callback: Callable):
        """TTS 완료 콜백 제거"""
        if callback in self.completion_callbacks:
            self.completion_callbacks.remove(callback)
            logger.debug(f"[TTS] 세션 {self.session_id[:4]}: 완료 콜백 제거")

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 가져오기 (lazy initialization)"""
        if self.http_session is None or self.http_session.closed:
            # TODO 이거 timeout 환경 변수로 뺴는 것을 고려
            timeout = aiohttp.ClientTimeout(total=30.0)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
        return self.http_session

    def _prepare_tts_payload(self, text: str) -> dict:
        """TTS 요청 페이로드 준비"""
        return {
            "input": text,
            "voice": self.voice,
            "pad_silence": self.pad_silence,
            "tempo": self.tempo,
            "gain_db": self.gain_db,
            "sample_rate": self.sample_rate,
            "return_type": self.return_type,
            "stream": self.stream,
            "response_format": self.response_format,
            "lang": self.lang,
            "cache": self.cache
        }

    async def process_text_to_speech(self, text: str) -> Optional[TTSResponse]:
        """텍스트를 TTS 서버로 처리 (큐 방식)"""
        if not text.strip():
            return None

        # 처리 중인 경우 큐에 추가
        if self.is_processing:
            if len(self.pending_requests) >= self.max_pending_requests:
                # 큐가 가득 찬 경우 오래된 요청 제거
                removed_request = self.pending_requests.pop(0)
                logger.warning(f"[TTS] 세션 {self.session_id[:4]}: 대기 큐 가득참, 오래된 요청 제거 - '{removed_request[:30]}...'")

            self.pending_requests.append(text)
            logger.info(f"[TTS] 세션 {self.session_id[:4]}: 처리 중이므로 요청 대기 (대기 수: {len(self.pending_requests)})")
            return None

        # 비동기적으로 처리 시작
        asyncio.create_task(self._process_request_queue(text))
        return None

    async def _process_request_queue(self, initial_text: str):
        """요청 큐를 순차적으로 처리"""
        async with self.request_semaphore:
            self.is_processing = True

            try:
                # 첫 번째 요청 처리
                await self._process_single_request(initial_text)

                # 대기 중인 요청들 순차 처리
                while self.pending_requests:
                    next_text = self.pending_requests.pop(0)
                    logger.info(
                        f"[TTS] 세션 {self.session_id[:4]}: 대기 요청 처리 - '{next_text[:30]}...' (남은 대기: {len(self.pending_requests)})")
                    await self._process_single_request(next_text)

                    # 처리 간 짧은 대기로 시스템 부하 방지
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"[TTS] 세션 {self.session_id[:4]}: 요청 큐 처리 중 오류 - {e}")
            finally:
                self.is_processing = False
                logger.info(f"[TTS] 세션 {self.session_id[:4]}: 요청 큐 처리 완료")

    def _prepare_llm_messages(self, text: str) -> List[Dict[str, str]]:
        messages = []

        # 시스템 프롬프트 추가
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 예시 추가
        messages.append({"role": "user", "content": "[질문]\n참 다행이네요!"})
        messages.append({"role": "assistant", "content": "happiness"})
        messages.append({"role": "user", "content": "[질문]\n그런 일이 벌어지다니, 참 안타까워"})
        messages.append({"role": "assistant", "content": "sadness"})
        messages.append({"role": "user", "content": "[질문]\n 진작에 말해주셨으면 제가 해결했을 텐데, 왜 "})
        messages.append({"role": "assistant", "content": "anger"})
        messages.append({"role": "user", "content": "[질문]\n걱정마세요, 당신이라면 충분히 할 수 있어요!"})
        messages.append({"role": "assistant", "content": "encourage"})
        messages.append({"role": "user", "content": "[질문]\n양자역학은 미시 세계의 물리적 현상을 설명하기 위해 등장한 이론으로, 원자와 기본 입자의 행동을 이해하는 데 핵심적인 역할을 합니다."})
        messages.append({"role": "assistant", "content": "neutral"})
        # 현재 사용자 입력 추가
        messages.append({"role": "user", "content": f"[질문]\n{text}"})

        return messages

    async def _process_single_request(self, text: str) -> Optional[TTSResponse]:
        """단일 TTS 요청 처리"""
        try:
            start_time = time.time()
            logger.info(f"🔊 [TTS] 세션 {self.session_id[:4]}: TTS 서버 요청 시작 - '{text[:50]}...'")

            messages = self._prepare_llm_messages(text)
            try:
                llm_response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_completion_tokens=10,
                    temperature=0.3,
                    stream=False,
                    timeout=1000.0,
                )
                affective_state = llm_response.choices[0].message.content.strip().split()[0].lower()
            except Exception as e:
                logger.warning(f"[TTS] 세션 {self.session_id[:4]}: LLM 호출 오류 - {e}")
                affective_state = "neutral"

            if affective_state in self.valid_affective_states:
                logger.info(f"[TTS] 세션 {self.session_id[:4]}: 감정상태 - {affective_state}")
            else:
                logger.warning(f"[TTS] 세션 {self.session_id[:4]}: 등록되지 않은 감정상태 - {affective_state}")
                affective_state = "neutral"

            # HTTP 세션 준비
            session = await self._get_http_session()

            # 요청 페이로드 준비
            payload = self._prepare_tts_payload(text)

            # TTS 서버에 POST 요청
            async with session.post(
                    self.endpoint,
                    json=payload,
                    headers={
                        'accept': 'application/json',
                        'Content-Type': 'application/json'
                    }
            ) as response:
                if response.status == 200:
                    # WAV 데이터 읽기
                    audio_data = await response.read()
                    processing_time = time.time() - start_time

                    if not audio_data:
                        logger.warning(f"[TTS] 세션 {self.session_id[:4]}: 빈 오디오 데이터 수신")
                        return None

                    # 응답 객체 생성
                    tts_response = TTSResponse(
                        original_text=text,
                        audio_data=audio_data,
                        timestamp=time.time(),
                        session_id=self.session_id,
                        processing_time=processing_time,
                        voice_id=self.voice,
                        sample_rate=self.sample_rate,
                        audio_format=self.response_format,
                        affective_state=affective_state,
                    )

                    # 결과 저장
                    self.processed_responses.append(tts_response)

                    formatted_time = time.strftime("%H:%M:%S", time.localtime(tts_response.timestamp))
                    logger.info(f"✅ [TTS-{self.session_id[:4]}] [{formatted_time}] 음성 생성 완료 "
                                f"({len(audio_data)} bytes, {processing_time:.2f}초)")

                    # ✅ 완료 콜백 즉시 호출 (폴링 대신!)
                    await self._notify_completion(tts_response)

                    return tts_response

                else:
                    # HTTP 오류 처리
                    error_text = await response.text()
                    logger.error(f"[TTS] 세션 {self.session_id[:4]}: TTS 서버 오류 {response.status} - {error_text}")
                    return None

        except asyncio.TimeoutError:
            logger.error(f"[TTS] 세션 {self.session_id[:4]}: TTS 서버 요청 타임아웃 - '{text[:50]}...'")
            return None

        except aiohttp.ClientError as e:
            logger.error(f"[TTS] 세션 {self.session_id[:4]}: TTS 서버 연결 오류 - {e}")
            return None

        except Exception as e:
            logger.error(f"[TTS] 세션 {self.session_id[:4]}: TTS 처리 오류 - {e}")
            return None

    # ✅ 완료 알림 메서드 추가
    async def _notify_completion(self, tts_response: TTSResponse):
        """TTS 완료를 즉시 알림"""
        if not self.completion_callbacks:
            return

        for callback in self.completion_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tts_response)
                else:
                    callback(tts_response)
            except Exception as e:
                logger.error(f"[TTS] 세션 {self.session_id[:4]}: 완료 콜백 오류 - {e}")

    async def cleanup(self):
        """TTS 프로세서 정리"""
        logger.info(f"[TTS] 세션 {self.session_id[:4]}: 정리 시작")

        try:
            # 처리 중인 작업 완료 대기 (최대 15초)
            wait_count = 0
            while self.is_processing and wait_count < 150:
                await asyncio.sleep(0.1)
                wait_count += 1

            if self.is_processing:
                logger.warning(f"[TTS] 세션 {self.session_id[:4]}: 처리 완료 대기 시간 초과")

            # 대기 중인 요청들 정리
            if self.pending_requests:
                logger.info(f"[TTS] 세션 {self.session_id[:4]}: {len(self.pending_requests)}개의 대기 요청 정리")
                self.pending_requests.clear()

            # ✅ 콜백 정리
            self.completion_callbacks.clear()

            # HTTP 세션 정리
            if self.http_session and not self.http_session.closed:
                await self.http_session.close()

        except Exception as e:
            logger.error(f"[TTS] 세션 {self.session_id[:4]}: 정리 중 오류 - {e}")


class TTSManager:
    """여러 세션의 TTS 프로세서 관리"""

    def __init__(self):
        self.processors: Dict[str, TTSProcessor] = {}

    def create_processor(self, session_id: str) -> TTSProcessor:
        """새 TTS 프로세서 생성"""
        if session_id in self.processors:
            logger.warning(f"[TTS] 프로세서가 이미 존재: {session_id[:8]}")
            return self.processors[session_id]

        processor = TTSProcessor(session_id)
        self.processors[session_id] = processor
        logger.info(f"[TTS] 새 프로세서 생성: {session_id[:8]}")
        return processor

    def get_processor(self, session_id: str) -> Optional[TTSProcessor]:
        """TTS 프로세서 가져오기"""
        return self.processors.get(session_id)

    async def process_text_to_speech(self, session_id: str, text: str) -> Optional[TTSResponse]:
        """특정 세션에서 텍스트를 음성으로 변환"""
        processor = self.get_processor(session_id)
        if not processor:
            logger.error(f"[TTS] 존재하지 않는 세션: {session_id[:8]}")
            return None

        return await processor.process_text_to_speech(text)

    async def remove_processor(self, session_id: str):
        """TTS 프로세서 제거"""
        if session_id in self.processors:
            await self.processors[session_id].cleanup()
            del self.processors[session_id]
            logger.info(f"[TTS] 프로세서 제거: {session_id[:8]}")

    async def cleanup_all(self):
        """모든 TTS 프로세서 정리"""
        cleanup_tasks = [processor.cleanup() for processor in self.processors.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.processors.clear()
        logger.info("[TTS] 모든 프로세서 정리 완료")


# 전역 TTS 매니저
tts_manager = TTSManager()
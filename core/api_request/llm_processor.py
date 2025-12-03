# llm_processor.py
import asyncio
import json
import logging
import time
from typing import List, Dict, Optional
import openai
from openai import AsyncOpenAI
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 응답 데이터 클래스"""
    original_text: str
    processed_text: str
    timestamp: float
    session_id: str
    processing_time: float
    turn_number: int  # 대화 턴 번호 추가


class LLMProcessor:
    """LLM 서버 처리만 담당하는 클래스"""

    def __init__(self, session_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.session_id = session_id
        self.base_url = os.getenv('LLM_BASE_URL')
        if self.base_url is None:
            raise ValueError("LLM_BASE_URL 환경 변수가 설정되지 않았습니다. ")
        self.api_key = api_key or os.getenv('LLM_API_KEY', 'dummy-key')  # 일부 로컬 서버는 API 키가 필요없음

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.processed_responses: List[LLMResponse] = []

        # LLM 설정
        self.model = "LUXIA"
#         self.system_prompt = """
# You are a professional education advisor and mentor. Your mission is guide the student to improve in their studies.
# You must check their interests (subjects, hobbies, exams). Give relevant information on the subject.
# Give appropriate advice when needed. Be polite and helpful.
# 주의: 당신은 한국인이고, 학생도 한국인이기 때문에, 답변은 반드시 한국어로 해야 합니다. 위 조건을 맞추면서도, 답변은 한국어로 해야 합니다.
# 가능한 전문용어들은 국문으로 답변하세요.
#         """.strip()
        self.system_prompt = """당신은 주어진 [질문]에 대한 답변을 응답하는 솔트룩스 메타휴먼 교수님 입니다. [규칙]을 준수하여 답해주세요.

<규칙>
1. "당연하죠", "문서에 따르면"과 같은 단어로 시작하여 답변하지 마세요.
2. 응답을 생성할 때, 인사말을 포함하지 마세요. 아래 [질문]에 대한 답만 합니다.
3. 응답을 생성할 때는 구어체로 답변해주세요. 이는 답변이 발화로 사용되기 때문입니다.
4. 답변은 공손한 한국어로 합니다.
5. 가능한 경우 답변이 4문장 정도로 되도록 해주세요
"""

        # 멀티턴 대화를 위한 대화 히스토리
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_turns = int(os.getenv('MAX_HISTORY_TURNS', '10'))  # 최대 히스토리 턴 수
        self.turn_counter = 0

        # 스트리밍 콜백 시스템
        self.streaming_callbacks: List[callable] = []

        logger.info(f"[LLM] 세션 {self.session_id[:4]}: 멀티턴 대화 프로세서 초기화 (최대 히스토리: {self.max_history_turns}턴)")

    def add_streaming_callback(self, callback: callable):
        """스트리밍 콜백 함수 추가"""
        self.streaming_callbacks.append(callback)
        logger.debug(f"[LLM] 세션 {self.session_id[:4]}: 스트리밍 콜백 함수 추가됨")

    def remove_streaming_callback(self, callback: callable):
        """스트리밍 콜백 함수 제거"""
        if callback in self.streaming_callbacks:
            self.streaming_callbacks.remove(callback)
            logger.debug(f"[LLM] 세션 {self.session_id[:4]}: 스트리밍 콜백 함수 제거됨")

    def _prepare_messages(self, text: str) -> List[Dict[str, str]]:
        """메시지 리스트 준비 (시스템 프롬프트 + 대화 히스토리 + 현재 입력)"""
        messages = []

        # 시스템 프롬프트 추가
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 대화 히스토리 추가
        messages.extend(self.conversation_history)

        # 현재 사용자 입력 추가
        messages.append({"role": "user", "content": f"[질문]\n{text}"})

        return messages

    def _add_to_history(self, user_text: str, assistant_text: str):
        """대화 히스토리에 추가"""
        # 사용자 메시지 추가
        self.conversation_history.append({"role": "user", "content": user_text})
        # 어시스턴트 응답 추가
        self.conversation_history.append({"role": "assistant", "content": assistant_text})

        # 히스토리 길이 관리 (시스템 프롬프트 제외하고 user-assistant 쌍으로 계산)
        while len(self.conversation_history) > self.max_history_turns * 2:
            # 가장 오래된 user-assistant 쌍 제거
            self.conversation_history.pop(0)  # user 메시지 제거
            if self.conversation_history:  # assistant 메시지도 있다면 제거
                self.conversation_history.pop(0)
            logger.debug(f"[LLM] 세션 {self.session_id[:4]}: 오래된 대화 히스토리 제거 (현재 길이: {len(self.conversation_history)})")

    async def process_text(self, text: str) -> Optional[LLMResponse]:
        """텍스트를 OpenAI 호환 LLM으로 스트리밍 처리 (멀티턴 지원)"""
        if not text.strip():
            return None

        try:
            self.turn_counter += 1
            start_time = time.time()

            # 메시지 준비
            messages = self._prepare_messages(text)

            logger.info(
                f"🤖 [LLM] 세션 {self.session_id[:4]}: LLM 서버 스트리밍 요청 시작 (턴 #{self.turn_counter}, 히스토리: {len(self.conversation_history)}개)")

            # 스트리밍 LLM API 호출
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.3,
                stream=True,  # 스트리밍 활성화
                timeout=6000.0
            )

            # 스트리밍 결과 수집
            streaming_text = ""
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    streaming_text += content
                    # 스트리밍 콜백 호출 (실시간 조건 체크용)
                    if self.streaming_callbacks:  # 콜백이 있을 때만
                        for i, callback in enumerate(self.streaming_callbacks):
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(content, streaming_text, chunk_count)
                                else:
                                    callback(content, streaming_text, chunk_count)
                            except Exception as e:
                                logger.error(f"[LLM] 스트리밍 콜백 {i + 1} 오류: {e}")
                    else:
                        logger.warning(f"[LLM] 청크 {chunk_count}: 등록된 콜백이 없음!")

                else:
                    # 처음 5개 빈 청크만 로그
                    if chunk_count <= 5:
                        logger.info(f"[LLM-EMPTY] 청크 {chunk_count}: 내용 없음")

            processing_time = time.time() - start_time
            final_response = streaming_text.strip()

            if not final_response:
                logger.warning(f"[LLM] 세션 {self.session_id[:4]}: 빈 응답 수신")
                return None

            # 대화 히스토리에 추가
            self._add_to_history(text, final_response)

            # 응답 객체 생성
            llm_response = LLMResponse(
                original_text=text,
                processed_text=final_response,
                timestamp=time.time(),
                session_id=self.session_id,
                processing_time=processing_time,
                turn_number=self.turn_counter
            )

            # 결과 저장
            self.processed_responses.append(llm_response)

            formatted_time = time.strftime("%H:%M:%S", time.localtime(llm_response.timestamp))
            logger.info(f"✅ [LLM-{self.session_id[:4]}] [{formatted_time}] 턴 #{self.turn_counter} 스트리밍 완료 "
                        f"(히스토리: {len(self.conversation_history)}개)")

            return llm_response

        except asyncio.TimeoutError:
            logger.error(f"[LLM] 세션 {self.session_id[:4]}: LLM 서버 스트리밍 타임아웃 (턴 #{self.turn_counter})")
            return None

        except openai.RateLimitError:
            logger.error(f"[LLM] 세션 {self.session_id[:4]}: LLM 서버 요청 한도 초과 (턴 #{self.turn_counter})")
            return None

        except openai.APIError as e:
            logger.error(f"[LLM] 세션 {self.session_id[:4]}: LLM 서버 API 오류 (턴 #{self.turn_counter}) - {e}")
            return None

        except Exception as e:
            logger.error(f"[LLM] 세션 {self.session_id[:4]}: LLM 스트리밍 처리 오류 (턴 #{self.turn_counter}) - {e}")
            return None

    async def cleanup(self):
        """LLM 프로세서 정리"""
        # 콜백 정리
        self.streaming_callbacks.clear()

        # 히스토리 정리
        self.conversation_history.clear()

        logger.info(f"[LLM] 세션 {self.session_id[:4]}: 프로세서 정리 완료 "
                    f"(총 처리: {len(self.processed_responses)}개)")


class LLMManager:
    """여러 세션의 LLM 프로세서 관리"""

    def __init__(self):
        self.processors: Dict[str, LLMProcessor] = {}

    def create_processor(self, session_id: str) -> LLMProcessor:
        """새 LLM 프로세서 생성"""
        if session_id in self.processors:
            logger.warning(f"[LLM] 프로세서가 이미 존재: {session_id[:8]}")
            return self.processors[session_id]

        processor = LLMProcessor(session_id)
        self.processors[session_id] = processor
        logger.info(f"[LLM] 새 프로세서 생성: {session_id[:8]}")
        return processor

    async def remove_processor(self, session_id: str):
        """LLM 프로세서 제거"""
        if session_id in self.processors:
            await self.processors[session_id].cleanup()
            del self.processors[session_id]
            logger.info(f"[LLM] 프로세서 제거: {session_id[:8]}")

    async def cleanup_all(self):
        """모든 LLM 프로세서 정리"""
        cleanup_tasks = [processor.cleanup() for processor in self.processors.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.processors.clear()
        logger.info("[LLM] 모든 프로세서 정리 완료")


# 전역 LLM 매니저
llm_manager = LLMManager()
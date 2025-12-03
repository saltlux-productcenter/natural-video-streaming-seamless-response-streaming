# stt_processor.py (순수 STT만 담당)
import asyncio
import json
import logging
import websockets
import time
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class STTProcessor:
    """STT 처리만 담당하는 독립적인 클래스"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.stt_uri = os.getenv('STT_URI')
        if self.stt_uri == "None":
            raise ValueError("STT_URI 환경 변수가 설정되지 않았습니다. ")
        self.ws = None
        self.is_connected = False
        self.stt_results: List[Dict] = []
        self.last_result = ""
        self.eos_sent = False

        # STT 결과 콜백 함수들
        self.result_callbacks: List[callable] = []

    def add_result_callback(self, callback: callable):
        """STT 결과 콜백 함수 추가"""
        self.result_callbacks.append(callback)
        logger.debug(f"[STT] 세션 {self.session_id[:4]}: 콜백 함수 추가됨")

    def remove_result_callback(self, callback: callable):
        """STT 결과 콜백 함수 제거"""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)
            logger.debug(f"[STT] 세션 {self.session_id[:4]}: 콜백 함수 제거됨")

    async def connect_stt(self) -> bool:
        """STT 서버에 연결"""
        try:
            self.ws = await websockets.connect(
                self.stt_uri,
                ping_interval=20,
                ping_timeout=5,
                close_timeout=5
            )

            self.is_connected = True
            self.eos_sent = False
            logger.info(f"[STT] 세션 {self.session_id[:4]}: STT 서버 연결 성공")

            # STT 결과 수신 태스크 시작
            asyncio.create_task(self._listen_stt_results())
            return True

        except Exception as e:
            logger.error(f"[STT] 세션 {self.session_id[:4]}: STT 연결 실패 - {e}")
            self.is_connected = False
            return False

    async def _listen_stt_results(self):
        """STT 결과 수신 및 콜백 호출"""
        try:
            async for message in self.ws:
                try:
                    # JSON 파싱
                    result = json.loads(message)

                    # transcript와 final 필드 처리
                    if 'transcript' in result:
                        transcript = result['transcript'].strip()
                        is_final = result.get('final', False)

                        if transcript:
                            await self._process_stt_result(transcript, result, is_final)

                except json.JSONDecodeError:
                    # JSON이 아닌 상태 메시지들은 debug 레벨로 로그
                    if message.strip():
                        status_messages = [
                            "Connected at Master", "Connected at Worker",
                            "No Available Workers", "Could not connect at Worker",
                            "Disconnected", "Reconnecting"
                        ]
                        if any(status in message for status in status_messages):
                            logger.debug(f"[STT] 세션 {self.session_id[:4]}: {message}")
                        else:
                            logger.warning(f"[STT] 세션 {self.session_id[:4]}: 알 수 없는 메시지 - {message}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[STT] 세션 {self.session_id[:4]}: 연결 종료됨")
            self.is_connected = False

        except Exception as e:
            logger.error(f"[STT] 세션 {self.session_id[:4]}: 수신 오류 - {e}")
            self.is_connected = False

    async def _process_stt_result(self, text: str, full_result: Dict, is_final: bool):
        """STT 결과 처리 및 콜백 호출"""
        timestamp = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.localtime(timestamp))

        # 결과 저장
        result_data = {
            'timestamp': timestamp,
            'text': text,
            'is_final': is_final,
            'full_result': full_result
        }

        # final 결과만 저장
        if is_final:
            self.stt_results.append(result_data)
            self.last_result = text

            # STT 로그 출력
            logger.info(f"🎤 [STT-{self.session_id[:4]}] [{formatted_time}] [FINAL] {text}")

            # 등록된 콜백 함수들 호출
            for callback in self.result_callbacks:
                try:
                    # 비동기 콜백 지원
                    if asyncio.iscoroutinefunction(callback):
                        await callback(text, is_final, result_data)
                    else:
                        callback(text, is_final, result_data)
                except Exception as e:
                    logger.error(f"[STT] 세션 {self.session_id[:4]}: 콜백 함수 실행 오류 - {e}")

        # 결과 수 제한
        if len(self.stt_results) > 100:
            self.stt_results = self.stt_results[-100:]

    async def send_audio_data(self, audio_data: bytes):
        """오디오 데이터를 STT 서버에 전송"""
        if self.is_connected and self.ws:
            try:
                await self.ws.send(audio_data)
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"[STT] 세션 {self.session_id[:4]}: 전송 중 연결 끊어짐")
                self.is_connected = False
            except Exception as e:
                logger.error(f"[STT] 세션 {self.session_id[:4]}: 오디오 전송 실패 - {e}")
                self.is_connected = False

    async def send_eos(self):
        """End of Speech 신호 전송"""
        if self.is_connected and self.ws and not self.eos_sent:
            try:
                await self.ws.send("EOS")
                self.eos_sent = True
                logger.info(f"[STT] 세션 {self.session_id[:4]}: EOS 전송됨")
            except Exception as e:
                logger.error(f"[STT] 세션 {self.session_id[:4]}: EOS 전송 실패 - {e}")

    async def close(self):
        """STT 연결 종료"""
        if self.ws:
            try:
                # EOS 전송
                await self.send_eos()
                await asyncio.sleep(0.1)

                # 연결 종료
                await self.ws.close()
                logger.info(f"[STT] 세션 {self.session_id[:4]}: 연결 종료 완료")
            except Exception as e:
                logger.error(f"[STT] 세션 {self.session_id[:4]}: 종료 중 오류 - {e}")

        self.is_connected = False
        self.ws = None
        self.result_callbacks.clear()
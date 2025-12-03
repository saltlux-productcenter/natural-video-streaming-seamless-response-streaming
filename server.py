# server.py
import asyncio
import json
import logging
import cv2
import uuid
import numpy as np
import base64
import fractions
import time
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, VideoStreamTrack, AudioStreamTrack, \
    RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamError
from core.pipeline_processor import pipeline_manager
from core.sync_manager import sync_manager_pool
from scipy.signal import resample_poly
import av
import os
from asyncio.queues import Queue

# 로그 설정
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
if log_level == 'INFO':
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s.%(msecs)03d- %(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
else:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

# 전역 세션 관리
sessions = {}


class AvatarVideoTrack(VideoStreamTrack):
    """AI 아바타 비디오 트랙 (25fps)"""

    def __init__(self, session_id):
        super().__init__()
        self.session_id = session_id

        # 현재 프레임
        self.current_frame = None

        # 대기 이미지
        self.idle_image = self._load_idle_image()
        self.bg_image = self._load_bg_image()
        self.mask = None

        # PTS
        self.frame_rate = 25
        self.time_base = fractions.Fraction(1, 25)
        self._timestamp = 0
        self._start = None

        # ✅ 로깅 관리
        self.frame_count = 0
        self.last_log_time = time.time()
        self.log_interval = 5.0  # 5초마다 로그
        self.idle_frame_count = 0
        self.lipsync_frame_count = 0
        self.last_frame_type = None  # 'idle' or 'lipsync'

        logger.info(f"[VIDEO] 세션 {session_id[:8]}: 비디오 트랙 생성 (25fps)")

    def _load_idle_image(self):
        """대기 이미지 로드"""
        try:
            idle_path = "static/idle_avatar.png"
            if os.path.exists(idle_path):
                img = cv2.imread(idle_path)
                if img is not None:
                    return img
        except Exception as e:
            logger.error(f"[VIDEO] 대기 이미지 로드 실패: {e}")

        # 기본 이미지
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "AI Avatar",
            (220, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )
        return img

    def _load_bg_image(self):
        """배경 이미지 로드"""
        try:
            bg_path = "static/bg_blur_image.png"
            if os.path.exists(bg_path):
                img = cv2.imread(bg_path)
                if img is not None:
                    return img
        except Exception as e:
            logger.error(f"[VIDEO] 대기 이미지 로드 실패: {e}")

        # 기본 이미지
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "AI Avatar",
            (220, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )
        return img

    async def recv(self):
        """25fps로 프레임 송출"""
        # Handle timestamps properly
        if self._start is not None:
            wait = self._start + (self._timestamp / self.frame_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
            wait = 0

        try:
            # 립싱크 프레임 또는 대기 이미지
            frame_type = "idle"
            if self.current_frame is not None:
                try:
                    img_bytes = base64.b64decode(self.current_frame)
                    img_array = cv2.imdecode(
                        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                    )

                    if img_array is None:
                        img_array = self.idle_image
                    else:
                        frame_type = "lipsync"
                except Exception as e:
                    logger.error(f"[VIDEO] 프레임 디코딩 오류: {e}")
                    img_array = self.idle_image
                    frame_type = "idle"
            else:
                img_array = self.idle_image

            # x: 928, y: 522
            curr_h, curr_w = img_array.shape[:2]
            new_h, new_w = 522, int(round(curr_w * 522 / curr_h))
            l_pad = (928 - new_w) // 2
            r_pad = (928 - l_pad - new_w)
            img_array = cv2.resize(
                img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC
            )
            img_array = cv2.copyMakeBorder(
                img_array,
                0, 0, l_pad, r_pad,
                cv2.BORDER_REPLICATE)
            bg_img_array = cv2.resize(self.bg_image, (928, 522), interpolation=cv2.INTER_CUBIC)
            if self.mask is None:
                mask = np.zeros((522, 928), dtype=np.float32)
                mask[:new_h, l_pad : l_pad + new_w] = 1.0
                mask = cv2.blur(mask, (50, 50))
                self.mask = mask[:,:,None]
            img_array = (img_array * self.mask + bg_img_array * (1-self.mask)).astype(np.uint8)

            # 프레임 카운트
            self.frame_count += 1
            if frame_type == "idle":
                self.idle_frame_count += 1
            else:
                self.lipsync_frame_count += 1

            # ✅ 상태 전환 감지 (즉시 로그)
            if frame_type != self.last_frame_type and self.last_frame_type is not None:
                if frame_type == "lipsync":
                    logger.info(
                        f"[VIDEO] 세션 {self.session_id[:4]}: 립싱크 프레임 재생 시작 (프레임#{self._timestamp})"
                    )
                else:
                    logger.info(
                        f"[VIDEO] 세션 {self.session_id[:4]}: 대기 이미지로 전환 (프레임#{self._timestamp})"
                    )

            self.last_frame_type = frame_type

            # VideoFrame 생성 (PTS 설정)
            video_frame = av.VideoFrame.from_ndarray(img_array, format="bgr24")
        except Exception as e:
            logger.error(
                f"[VIDEO-ERROR] 비디오 트랙 오류 (프레임#{self._timestamp}): {e}"
            )
            video_frame = av.VideoFrame.from_ndarray(self.idle_image, format="bgr24")

        video_frame.pts = self._timestamp
        video_frame.time_base = self.time_base
        self._timestamp += max(1, (wait // self.frame_rate))
        return video_frame


    def play_synced_frame(self, frame_data: str):
        """립싱크 프레임 수신"""
        self.current_frame = frame_data
        logger.debug(f"[VIDEO-RECEIVE] 세션 {self.session_id[:4]}: 새로운 립싱크 프레임 수신 (크기: {len(frame_data)} bytes)")

    def force_idle(self):
        """대기 상태로 전환"""
        if self.current_frame is not None:
            logger.info(f"[VIDEO] 세션 {self.session_id[:4]}: 강제 대기 상태 전환")
        self.current_frame = None


class AvatarAudioTrack(AudioStreamTrack):
    """AI 아바타 오디오 트랙 (48kHz)"""

    def __init__(self, session_id):
        super().__init__()
        self.session_id = session_id

        # STT 처리용
        self.audio_buffer = bytearray()
        self.buffer_size = 1024
        self.is_ended = False

        # TTS 재생용
        self.tts_audio_queue = Queue()
        self.current_tts_frames = Queue()

        # 오디오 설정
        self.sample_rate = 48000
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.frame_samples = int(self.sample_rate * 0.02)  # 20ms

        # PTS
        self._start = None
        self._timestamp = 0

        logger.info(f"[AUDIO] 세션 {session_id[:8]}: 오디오 트랙 생성 (48kHz)")

    async def recv(self):
        """20ms 간격으로 오디오 프레임 송출"""
        # Handle timestamps properly
        if self._start is not None:
            wait = self._start + (self._timestamp / self.sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        try:
            audio_frame = await self._get_next_tts_frame()
        except Exception as e:
            logger.error(f"[AUDIO] 오디오 트랙 오류: {e}")
            audio_frame = self._generate_silence()
        audio_frame.time_base = self.time_base
        audio_frame.rate = self.sample_rate
        audio_frame.pts = self._timestamp
        self._timestamp += self.frame_samples
        return audio_frame

    async def _get_next_tts_frame(self):
        """다음 TTS 프레임 가져오기"""
        # 현재 재생 중인 프레임

        if not self.current_tts_frames.empty():
            frame = await self.current_tts_frames.get()
            return frame

        # 큐에서 다음 WAV 가져오기
        if not self.tts_audio_queue.empty():
            next_wav_data = await self.tts_audio_queue.get()

            for frame in self._convert_wav_to_frames(next_wav_data):
                await self.current_tts_frames.put(frame)

            if not self.current_tts_frames.empty():
                frame = await self.current_tts_frames.get()
                return frame

        return self._generate_silence()

    async def play_synced_audio(self, wav_data: bytes):
        """TTS 오디오 큐에 추가"""
        try:
            await self.tts_audio_queue.put(wav_data)
        except Exception as e:
            logger.error(f"[AUDIO] TTS 오디오 큐 추가 실패: {e}")

    def _convert_wav_to_frames(self, wav_data: bytes):
        """WAV를 20ms AudioFrame으로 변환"""
        try:
            if len(wav_data) < 44:
                return []

            channels = int.from_bytes(wav_data[22:24], byteorder='little')
            original_sample_rate = int.from_bytes(wav_data[24:28], byteorder='little')
            audio_data = wav_data[44:]
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # 모노로 변환
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)

            # 리샘플링 (24kHz -> 48kHz)
            if original_sample_rate != self.sample_rate:
                if original_sample_rate == 24000:
                    audio_array = resample_poly(audio_array, up=2, down=1).astype(np.int16)
                elif original_sample_rate == 22050:
                    audio_array = resample_poly(audio_array, up=160, down=147).astype(np.int16)
                else:
                    from scipy.signal import resample
                    new_length = int(len(audio_array) * self.sample_rate / original_sample_rate)
                    audio_array = resample(audio_array, new_length).astype(np.int16)

            frames = []
            for i in range(0, len(audio_array), self.frame_samples):
                chunk = audio_array[i:i + self.frame_samples]

                if len(chunk) < self.frame_samples:
                    chunk = np.pad(chunk, (0, self.frame_samples - len(chunk)), 'constant')

                audio_frame = av.AudioFrame.from_ndarray(
                    chunk.reshape(1, -1), format='s16', layout='mono'
                )
                frames.append(audio_frame)

            return frames

        except Exception as e:
            logger.error(f"[AUDIO] WAV 변환 오류: {e}")
            return []

    def _generate_silence(self):
        """무음 프레임 생성"""
        silence = np.zeros(self.frame_samples, dtype=np.int16)
        audio_frame = av.AudioFrame.from_ndarray(
            silence.reshape(1, -1), format='s16', layout='mono'
        )
        return audio_frame

    async def force_idle(self):
        """대기 상태로 전환"""
        while not self.tts_audio_queue.empty():
            await self.tts_audio_queue.get()
        while not self.current_tts_frames.empty():
            await self.current_tts_frames.get()


class ClientAudioSender:
    def __init__(self, session_id, audio_track, pipeline):
        super().__init__()
        self.session_id = session_id
        self.audio_track = audio_track
        self.pipeline = pipeline

        self.audio_buffer = bytearray()
        self.buffer_size = 1024
        self.is_ended = False

        # 오디오 설정
        self.sample_rate = 48000
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.frame_samples = int(self.sample_rate * 0.02)  # 20ms

        # PTS
        self._start = None
        self._timestamp = 0

    async def stop(self):
        self.is_ended = True

    async def run(self):
        while not self.is_ended and (self.audio_track.readyState == "live"):
            try:
                if self._start is not None:
                    wait = self._start + (self._timestamp / self.sample_rate) - time.time()
                    if wait > 0:
                        await asyncio.sleep(wait)
                else:
                    self._start = time.time()
                    self._timestamp = 0

                input_frame = await self.audio_track.recv()

                if self.pipeline.stt_processor.is_connected and not self.is_ended:
                    audio_data = self._convert_for_stt(input_frame)
                    if audio_data:
                        self.audio_buffer.extend(audio_data)

                        while len(self.audio_buffer) >= self.buffer_size:
                            chunk = bytes(self.audio_buffer[:self.buffer_size])
                            self.audio_buffer = self.audio_buffer[self.buffer_size:]
                            await self.pipeline.stt_processor.send_audio_data(chunk)
            except MediaStreamError:
                logger.debug("[AUDIO] 입력 오디오 종료")
                self.is_ended = True
            except Exception as e:
                logger.warning(f"[AUDIO] 입력 오디오 처리 오류: {e}")
                self.is_ended = True

    def _convert_for_stt(self, frame):
        """입력 오디오를 STT용으로 변환"""
        try:
            audio_array = frame.to_ndarray()
            original_sample_rate = frame.sample_rate

            if audio_array.shape[1] > 960:
                audio_array = audio_array.reshape(1, 960, 2).mean(axis=2).astype(np.int16)

            audio_array = audio_array[0]

            if audio_array.dtype != np.int16:
                audio_array = (audio_array * 32767).astype(np.int16)

            if original_sample_rate == 48000:
                audio_array = resample_poly(audio_array, up=1, down=3)
            else:
                step = int(original_sample_rate / 16000)
                audio_array = audio_array[::step]

            return audio_array.astype(np.int16).tobytes()

        except Exception as e:
            logger.error(f"[AUDIO] STT 변환 오류: {e}")
            return None

    async def _send_final_data(self):
        """종료시 STT EOS 전송"""
        try:
            if len(self.audio_buffer) > 0:
                self.audio_buffer.clear()
            if self.pipeline.stt_processor.is_connected:
                await self.pipeline.stt_processor.send_eos()
        except Exception as e:
            logger.error(f"[AUDIO] 종료 처리 오류: {e}")



class WebRTCSession:
    """WebRTC 세션"""

    def __init__(self, session_id):
        self.session_id = session_id

        # WebRTC 설정
        ice_servers = [
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"])
        ]

        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )

        # 파이프라인 및 동기화
        self.pipeline = pipeline_manager.create_pipeline(session_id)
        self.sync_manager = sync_manager_pool.create_manager(session_id)

        # 트랙
        self.avatar_video_track = None
        self.avatar_audio_track = None
        self.client_audio_sender = None

        # Data Channel
        self.data_channel = None

        # ICE candidates
        self.ice_candidates = []

        self._setup_event_handlers()
        self._setup_callbacks()

        logger.info(f"[SESSION] 세션 {session_id[:8]}: 생성")

    def _setup_event_handlers(self):
        """이벤트 핸들러 설정"""

        @self.pc.on("icecandidate")
        def on_icecandidate(candidate):
            if candidate:
                self.ice_candidates.append({
                    "candidate": candidate.sdp,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex
                })

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"[SESSION] 세션 {self.session_id[:4]}: 연결 상태 - {self.pc.connectionState}")
            if self.pc.connectionState in ["failed", "closed"]:
                await self.cleanup()

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"[SESSION] 세션 {self.session_id[:4]}: Data Channel 수신")
            self.data_channel = channel

            @channel.on("open")
            def on_open():
                logger.info(f"[SESSION] 세션 {self.session_id[:4]}: Data Channel 열림")

            @channel.on("close")
            def on_close():
                logger.info(f"[SESSION] 세션 {self.session_id[:4]}: Data Channel 닫힘")

        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"[SESSION] 세션 {self.session_id[:4]}: 트랙 수신 - {track.kind}")
            if track.kind == "audio":
                asyncio.create_task(self.pipeline.start_stt())
                self.client_audio_sender = ClientAudioSender(self.session_id, track, self.pipeline)
                asyncio.create_task(self.client_audio_sender.run())

        self.avatar_video_track = AvatarVideoTrack(self.session_id)
        self.pc.addTrack(self.avatar_video_track)
        self.avatar_audio_track = AvatarAudioTrack(self.session_id)
        self.pc.addTrack(self.avatar_audio_track)


    def _setup_callbacks(self):
        """콜백 설정"""

        # 동기화 콜백
        async def on_synced_audio(audio_data: bytes):
            if self.avatar_audio_track:
                await self.avatar_audio_track.play_synced_audio(audio_data)

        def on_synced_video(lipsync_frame: str):
            if self.avatar_video_track:
                self.avatar_video_track.play_synced_frame(lipsync_frame)

        async def on_idle():
            if self.avatar_audio_track:
                await self.avatar_audio_track.force_idle()
            if self.avatar_video_track:
                self.avatar_video_track.force_idle()

        self.sync_manager.set_callbacks(on_synced_audio, on_synced_video, on_idle)

        # 파이프라인 콜백
        def on_tts_complete(sentence_num: int, tts_response):
            asyncio.create_task(
                self.sync_manager.add_tts_response(sentence_num, tts_response)
            )

        def on_lipsync_frame(sentence_num: int, frame_data: str):
            asyncio.create_task(
                self.sync_manager.add_lipsync_frame(sentence_num, frame_data)
            )

        def on_lipsync_complete(sentence_num: int):
            asyncio.create_task(
                self.sync_manager.mark_lipsync_complete(sentence_num)
            )

        # STT 결과 콜백
        def on_stt_result(text: str, is_final: bool, result_data: dict):
            if self.data_channel and self.data_channel.readyState == "open":
                try:
                    message = {
                        "type": "stt_result",
                        "text": text,
                        "is_final": is_final,
                        "timestamp": time.time(),
                        "session_id": self.session_id[:4]
                    }

                    self.data_channel.send(json.dumps(message))
                    logger.info(f"[DATA-CHANNEL] STT: {text} (final={is_final})")

                except Exception as e:
                    logger.error(f"[DATA-CHANNEL] 전송 실패: {e}")

        self.pipeline.add_tts_complete_callback(on_tts_complete)
        self.pipeline.add_lipsync_frame_callback(on_lipsync_frame)
        self.pipeline.add_lipsync_complete_callback(on_lipsync_complete)
        self.pipeline.stt_processor.add_result_callback(on_stt_result)

    async def handle_offer(self, offer_data):
        """Offer 처리"""
        try:
            offer = RTCSessionDescription(sdp=offer_data["sdp"], type=offer_data["type"])

            if self.pc.signalingState != "stable":
                raise Exception(f"Invalid signaling state: {self.pc.signalingState}")

            await self.pc.setRemoteDescription(offer)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            # ICE gathering 대기
            for i in range(30):
                await asyncio.sleep(0.1)
                if self.pc.iceGatheringState == "complete" or len(self.ice_candidates) > 0:
                    break

            return {
                "sdp": self.pc.localDescription.sdp,
                "type": self.pc.localDescription.type,
                "session_id": self.session_id
            }

        except Exception as e:
            logger.error(f"[SESSION] Offer 처리 실패: {e}")
            raise e

    async def add_ice_candidate(self, candidate_data):
        """ICE candidate 추가"""
        try:
            candidate_str = candidate_data["candidate"]
            parts = candidate_str.split()

            if len(parts) < 8:
                return

            foundation = parts[0].split(':')[1] if ':' in parts[0] else parts[0]
            component = int(parts[1])
            protocol = parts[2]
            priority = int(parts[3])
            ip = parts[4]
            port = int(parts[5])

            typ = "host"
            if "typ" in parts:
                typ_index = parts.index("typ")
                if typ_index + 1 < len(parts):
                    typ = parts[typ_index + 1]

            candidate = RTCIceCandidate(
                component=component,
                foundation=foundation,
                ip=ip,
                port=port,
                priority=priority,
                protocol=protocol,
                type=typ,
                sdpMid=candidate_data["sdpMid"],
                sdpMLineIndex=candidate_data["sdpMLineIndex"]
            )

            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"[SESSION] ICE candidate 추가 실패: {e}")

    def get_ice_candidates(self):
        """ICE candidates 반환"""
        candidates = self.ice_candidates.copy()
        self.ice_candidates.clear()
        return candidates

    async def cleanup(self):
        """세션 정리"""
        try:
            logger.info(f"[SESSION] 세션 {self.session_id[:4]}: 정리")

            await sync_manager_pool.remove_manager(self.session_id)
            await pipeline_manager.remove_pipeline(self.session_id)
            # await self.pipeline.cleanup()

            if self.pc.connectionState != "closed":
                await self.pc.close()

            if self.session_id in sessions:
                del sessions[self.session_id]

            if self.client_audio_sender is not None:
                await self.client_audio_sender.stop()

        except Exception as e:
            logger.error(f"[SESSION] 정리 오류: {e}")


# API 엔드포인트
async def create_session(request):
    """세션 생성"""
    session_id = str(uuid.uuid4())
    session = WebRTCSession(session_id)
    sessions[session_id] = session

    logger.info(f"[API] 세션 생성: {session_id[:8]} (총 {len(sessions)}개)")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"session_id": session_id}),
        headers={"Access-Control-Allow-Origin": "*"}
    )


async def handle_offer(request):
    """Offer 처리"""
    try:
        data = await request.json()
        session_id = data.get("session_id")

        if not session_id or session_id not in sessions:
            raise Exception("Invalid session")

        session = sessions[session_id]
        answer_data = await session.handle_offer(data)

        return web.Response(
            content_type="application/json",
            text=json.dumps(answer_data),
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except Exception as e:
        logger.error(f"[API] Offer 처리 오류: {e}")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": str(e)}),
            headers={"Access-Control-Allow-Origin": "*"}
        )


async def handle_ice_candidate(request):
    """ICE candidate 처리"""
    try:
        data = await request.json()
        session_id = data.get("session_id")

        if not session_id or session_id not in sessions:
            raise Exception("Invalid session")

        session = sessions[session_id]
        await session.add_ice_candidate(data)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "ok"}),
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"[API] ICE candidate 처리 오류: {e}")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": str(e)}),
            headers={"Access-Control-Allow-Origin": "*"}
        )


async def get_ice_candidates(request):
    """ICE candidates 조회"""
    try:
        session_id = request.match_info['session_id']

        if session_id not in sessions:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"candidates": []}),
                headers={"Access-Control-Allow-Origin": "*"}
            )

        session = sessions[session_id]
        candidates = session.get_ice_candidates()

        return web.Response(
            content_type="application/json",
            text=json.dumps({"candidates": candidates}),
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"[API] ICE candidates 조회 오류: {e}")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": str(e)}),
            headers={"Access-Control-Allow-Origin": "*"}
        )


async def get_sessions_info(request):
    """세션 정보 조회"""
    session_info = []
    for sid, session in sessions.items():
        pipeline_stats = session.pipeline.get_pipeline_stats()
        sync_stats = session.sync_manager.get_status()

        session_data = {
            "session_id": sid,
            "short_id": sid[:8],
            "connection_state": session.pc.connectionState,
            "sentence_count": pipeline_stats['sentence_count'],
            "is_processing": pipeline_stats['is_processing'],
            "sync_status": sync_stats,
            "data_channel_open": session.data_channel.readyState == "open" if session.data_channel else False
        }

        session_info.append(session_data)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "active_sessions": len(sessions),
            "sessions": session_info
        }, ensure_ascii=False, indent=2),
        headers={"Access-Control-Allow-Origin": "*"}
    )


async def index(request):
    """메인 페이지"""
    try:
        with open("static/fullscreen_display.html", "r", encoding="utf-8") as f:
            content = f.read()
        return web.Response(content_type="text/html", text=content)
    except FileNotFoundError:
        return web.Response(text="HTML 파일을 찾을 수 없습니다", status=404)


async def options_handler(request):
    """CORS 처리"""
    return web.Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )


async def cleanup_on_shutdown(app):
    """서버 종료시 정리"""
    logger.info("[SERVER] 서버 종료 중...")

    cleanup_tasks = [session.cleanup() for session in sessions.values()]
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    sessions.clear()

    await pipeline_manager.cleanup_all()
    await sync_manager_pool.cleanup_all()

    logger.info("[SERVER] 서버 종료 완료")


def create_app():
    app = web.Application()

    # 라우터 설정
    app.router.add_post("/create-session", create_session)
    app.router.add_post("/offer", handle_offer)
    app.router.add_post("/ice-candidate", handle_ice_candidate)
    app.router.add_get("/ice-candidates/{session_id}", get_ice_candidates)
    app.router.add_get("/sessions", get_sessions_info)
    app.router.add_get("/", index)

    # 정적 파일
    if os.path.exists("static"):
        app.router.add_static("/static/", path="static", name="static")

    # CORS
    app.router.add_options("/create-session", options_handler)
    app.router.add_options("/offer", options_handler)
    app.router.add_options("/ice-candidate", options_handler)

    # 종료 핸들러
    app.on_shutdown.append(cleanup_on_shutdown)

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("=" * 60)
    logger.info("AI Avatar WebRTC 서버 시작")
    logger.info("=" * 60)
    web.run_app(app, host="0.0.0.0", port=8080)
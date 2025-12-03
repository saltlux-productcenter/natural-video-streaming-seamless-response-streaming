import cv2
import numpy as np
from typing import List, Tuple


class FaceDetector:
    def __init__(self):
        # OpenCV의 Haar Cascade 사용
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 또는 MediaPipe 사용 (더 정확함)
        # import mediapipe as mp
        # self.mp_face_detection = mp.solutions.face_detection
        # self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def detect_faces_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        OpenCV Haar Cascade를 사용한 얼굴 감지

        Args:
            frame: 입력 이미지 프레임

        Returns:
            List of (x, y, width, height) tuples for detected faces
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # OpenCV 버전에 따른 반환 타입 처리
            if isinstance(faces, tuple):
                # 일부 OpenCV 버전에서는 tuple로 반환
                faces = faces[0] if len(faces) > 0 else []

            # numpy array를 list로 변환
            if hasattr(faces, 'tolist'):
                return faces.tolist()
            elif isinstance(faces, (list, tuple)):
                return list(faces)
            else:
                return []

        except Exception as e:
            print(f"얼굴 감지 중 오류 발생: {e}")
            return []

    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[dict]:
        """
        MediaPipe를 사용한 얼굴 감지 (주석 해제 후 사용)

        Args:
            frame: 입력 이미지 프레임

        Returns:
            List of face detection results with bounding boxes and confidence
        """
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = self.face_detection.process(rgb_frame)
        #
        # faces = []
        # if results.detections:
        #     for detection in results.detections:
        #         bbox = detection.location_data.relative_bounding_box
        #         h, w, _ = frame.shape
        #         faces.append({
        #             'x': int(bbox.xmin * w),
        #             'y': int(bbox.ymin * h),
        #             'width': int(bbox.width * w),
        #             'height': int(bbox.height * h),
        #             'confidence': detection.score[0]
        #         })
        # return faces
        pass

    def draw_face_boxes(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        감지된 얼굴 주위에 박스 그리기

        Args:
            frame: 원본 프레임
            faces: 감지된 얼굴 좌표 리스트

        Returns:
            박스가 그려진 프레임
        """
        try:
            result_frame = frame.copy()

            if not faces:
                return result_frame

            for face in faces:
                # face가 tuple이나 list인지 확인
                if isinstance(face, (tuple, list)) and len(face) >= 4:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

                    # 좌표 유효성 검사
                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        # 얼굴 주위에 초록색 박스 그리기
                        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # 얼굴 개수 표시
                        cv2.putText(result_frame, 'Face', (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 총 얼굴 개수 표시
            if faces:
                cv2.putText(result_frame, f'Total Faces: {len(faces)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return result_frame

        except Exception as e:
            print(f"얼굴 박스 그리기 중 오류: {e}")
            return frame

    def extract_face_regions(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        감지된 얼굴 영역을 추출

        Args:
            frame: 원본 프레임
            faces: 감지된 얼굴 좌표 리스트

        Returns:
            추출된 얼굴 이미지들의 리스트
        """
        face_images = []
        try:
            for face in faces:
                if isinstance(face, (tuple, list)) and len(face) >= 4:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

                    # 여유 공간을 두고 얼굴 영역 추출
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    if x2 > x1 and y2 > y1:  # 유효한 영역인지 확인
                        face_image = frame[y1:y2, x1:x2]
                        if face_image.size > 0:  # 빈 이미지가 아닌지 확인
                            face_images.append(face_image)
        except Exception as e:
            print(f"얼굴 영역 추출 중 오류: {e}")

        return face_images
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    box: tuple[int, int, int, int]
    detection_confidence: float
    quality: float
    embedding: np.ndarray


class FaceEngine:
    def __init__(self, yunet_path: Path, sface_path: Path, detection_threshold: float) -> None:
        if not yunet_path.exists() or not sface_path.exists():
            raise RuntimeError("Face model files are missing; rebuild the Docker image")
        self.detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), detection_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

    @staticmethod
    def decode(payload: bytes) -> np.ndarray:
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("unsupported_or_corrupt_image")
        return image

    @staticmethod
    def _quality(image: np.ndarray, face: np.ndarray, aligned: np.ndarray) -> float:
        height, width = image.shape[:2]
        _, _, w, h = [float(value) for value in face[:4]]
        face_ratio = min(1.0, (w * h) / max(1.0, width * height * 0.12))
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = min(1.0, blur_variance / 180.0)
        right_eye = np.array(face[4:6], dtype=np.float32)
        left_eye = np.array(face[6:8], dtype=np.float32)
        nose = np.array(face[8:10], dtype=np.float32)
        eye_mid = (right_eye + left_eye) / 2.0
        eye_distance = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
        nose_offset = float(np.linalg.norm(nose - eye_mid)) / eye_distance
        pose_score = max(0.0, 1.0 - abs(nose_offset - 0.65) / 0.75)
        brightness = float(np.mean(gray)) / 255.0
        exposure = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
        return float(np.clip(0.35 * sharpness + 0.30 * face_ratio + 0.20 * pose_score + 0.15 * exposure, 0, 1))

    def analyze(self, image: np.ndarray) -> list[FaceObservation]:
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        observations: list[FaceObservation] = []
        for face in faces:
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm == 0:
                continue
            feature /= norm
            x, y, w, h = [int(round(value)) for value in face[:4]]
            observations.append(FaceObservation(
                box=(max(0, x), max(0, y), max(0, w), max(0, h)),
                detection_confidence=float(face[-1]),
                quality=self._quality(image, face, aligned),
                embedding=feature,
            ))
        return observations

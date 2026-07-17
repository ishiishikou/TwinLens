from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("TWINLENS_DATA_DIR", "/data"))
    model_dir: Path = Path(os.getenv("TWINLENS_MODEL_DIR", "/models"))
    api_token: str = os.getenv("TWINLENS_API_TOKEN", "")
    fernet_key: str = os.getenv("TWINLENS_FERNET_KEY", "")
    max_upload_bytes: int = _int("TWINLENS_MAX_UPLOAD_BYTES", 12 * 1024 * 1024)
    min_detection_confidence: float = _float("TWINLENS_MIN_DETECTION_CONFIDENCE", 0.85)
    min_quality: float = _float("TWINLENS_MIN_QUALITY", 0.35)
    other_threshold: float = _float("TWINLENS_OTHER_THRESHOLD", 0.28)
    accept_threshold: float = _float("TWINLENS_ACCEPT_THRESHOLD", 0.42)
    margin_threshold: float = _float("TWINLENS_MARGIN_THRESHOLD", 0.08)
    min_references_per_subject: int = _int("TWINLENS_MIN_REFERENCES_PER_SUBJECT", 3)
    correction_retention_days: int = _int("TWINLENS_CORRECTION_RETENTION_DAYS", 30)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "twinlens.sqlite3"

    @property
    def yunet_path(self) -> Path:
        return self.model_dir / "face_detection_yunet_2023mar.onnx"

    @property
    def sface_path(self) -> Path:
        return self.model_dir / "face_recognition_sface_2021dec.onnx"

    def validate(self) -> None:
        if not self.api_token:
            raise RuntimeError("TWINLENS_API_TOKEN is required")
        if not self.fernet_key:
            raise RuntimeError("TWINLENS_FERNET_KEY is required")
        if self.other_threshold >= self.accept_threshold:
            raise RuntimeError("TWINLENS_OTHER_THRESHOLD must be lower than TWINLENS_ACCEPT_THRESHOLD")
        if self.margin_threshold <= 0:
            raise RuntimeError("TWINLENS_MARGIN_THRESHOLD must be positive")

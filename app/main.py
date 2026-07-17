from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings
from app.database import Database
from app.decision import DecisionLabel, aggregate_similarity, decide
from app.face_engine import FaceEngine
from app.security import VectorCipher, valid_api_token


class Subject(StrEnum):
    A = "A"
    B = "B"


class CorrectionLabel(StrEnum):
    A = "A"
    B = "B"
    INDETERMINATE = "INDETERMINATE"
    OTHER = "OTHER"


class CorrectionRequest(BaseModel):
    prediction_id: str = Field(min_length=8, max_length=100)
    corrected_label: CorrectionLabel
    promote_to_reference: bool = True


settings = Settings()
db = Database(settings.db_path)
cipher: VectorCipher
engine: FaceEngine


@asynccontextmanager
async def lifespan(_: FastAPI):
    global cipher, engine
    settings.validate()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db.initialize()
    cipher = VectorCipher(settings.fernet_key)
    engine = FaceEngine(settings.yunet_path, settings.sface_path, settings.min_detection_confidence)
    db.purge_old_predictions(settings.correction_retention_days)
    yield


app = FastAPI(title="TwinLens", version="0.1.0", lifespan=lifespan)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def require_token(x_twinlens_token: str | None = Header(default=None)) -> None:
    if not valid_api_token(settings.api_token, x_twinlens_token):
        raise HTTPException(status_code=401, detail="invalid_api_token")


async def read_image(file: UploadFile) -> tuple[bytes, np.ndarray]:
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="supported_types_are_jpeg_png_webp")
    payload = await file.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image_too_large")
    try:
        return payload, engine.decode(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def load_reference_vectors(subject: str) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for row in db.list_embeddings(subject):
        raw = cipher.decrypt(row["vector"])
        vectors.append(np.frombuffer(raw, dtype=np.float32).copy())
    return vectors


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": "OpenCV YuNet + SFace", "stores_original_images": False}


@app.get("/api/v1/stats", dependencies=[Depends(require_token)])
def stats() -> dict[str, object]:
    return {"references": db.counts(), "thresholds": {
        "other": settings.other_threshold,
        "accept": settings.accept_threshold,
        "margin": settings.margin_threshold,
        "minimum_quality": settings.min_quality,
    }}


@app.post("/api/v1/enroll/{subject}", dependencies=[Depends(require_token)])
async def enroll(subject: Subject, files: list[UploadFile] = File(...)) -> dict[str, object]:
    if not 1 <= len(files) <= 100:
        raise HTTPException(status_code=400, detail="upload_between_1_and_100_images")
    accepted = 0
    rejected: list[dict[str, str]] = []
    for file in files:
        payload, image = await read_image(file)
        observations = engine.analyze(image)
        if len(observations) != 1:
            rejected.append({"file": file.filename or "unnamed", "reason": "enrollment_requires_exactly_one_face"})
            continue
        observation = observations[0]
        if observation.quality < settings.min_quality:
            rejected.append({"file": file.filename or "unnamed", "reason": "low_image_quality"})
            continue
        digest = hashlib.sha256(payload).hexdigest()
        inserted = db.add_embedding(
            subject=subject.value,
            vector=cipher.encrypt(observation.embedding.tobytes()),
            quality=observation.quality,
            detection_confidence=observation.detection_confidence,
            source_sha256=digest,
        )
        if inserted:
            accepted += 1
        else:
            rejected.append({"file": file.filename or "unnamed", "reason": "duplicate_image"})
    return {"subject": subject.value, "accepted": accepted, "rejected": rejected, "references": db.counts()}


@app.post("/api/v1/identify", dependencies=[Depends(require_token)])
async def identify(file: UploadFile = File(...)) -> dict[str, object]:
    _, image = await read_image(file)
    observations = engine.analyze(image)
    if not observations:
        return {"faces": [], "overall": DecisionLabel.OTHER_OR_NO_FACE, "message": "no_face_detected"}
    references_a = load_reference_vectors("A")
    references_b = load_reference_vectors("B")
    counts = {"A": len(references_a), "B": len(references_b)}
    results: list[dict[str, object]] = []
    for index, observation in enumerate(observations):
        score_a = aggregate_similarity(observation.embedding, references_a)
        score_b = aggregate_similarity(observation.embedding, references_b)
        outcome = decide(
            score_a=score_a,
            score_b=score_b,
            quality=observation.quality,
            detection_confidence=observation.detection_confidence,
            count_a=counts["A"],
            count_b=counts["B"],
            min_quality=settings.min_quality,
            min_detection_confidence=settings.min_detection_confidence,
            min_references_per_subject=settings.min_references_per_subject,
            other_threshold=settings.other_threshold,
            accept_threshold=settings.accept_threshold,
            margin_threshold=settings.margin_threshold,
        )
        prediction_id = str(uuid.uuid4())
        db.save_prediction(
            prediction_id=prediction_id,
            vector=cipher.encrypt(observation.embedding.tobytes()),
            predicted_label=outcome.label,
            score_a=score_a,
            score_b=score_b,
            quality=observation.quality,
            detection_confidence=observation.detection_confidence,
        )
        results.append({
            "face_index": index,
            "prediction_id": prediction_id,
            "label": outcome.label,
            "box": observation.box,
            "score_a": round(score_a, 4),
            "score_b": round(score_b, 4),
            "margin": round(outcome.margin, 4),
            "quality": round(observation.quality, 4),
            "detection_confidence": round(observation.detection_confidence, 4),
            "reason": outcome.reason,
        })
    return {"faces": results, "overall": "MULTIPLE_FACES" if len(results) > 1 else results[0]["label"]}


@app.post("/api/v1/corrections", dependencies=[Depends(require_token)])
def correct(request: CorrectionRequest) -> dict[str, object]:
    prediction = db.get_prediction(request.prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="prediction_not_found_or_expired")
    db.correct_prediction(request.prediction_id, request.corrected_label)
    promoted = False
    if request.promote_to_reference and request.corrected_label in {CorrectionLabel.A, CorrectionLabel.B}:
        if float(prediction["quality"]) < settings.min_quality:
            raise HTTPException(status_code=400, detail="low_quality_prediction_cannot_be_promoted")
        digest = hashlib.sha256(f"correction:{request.prediction_id}".encode()).hexdigest()
        promoted = db.add_embedding(
            subject=request.corrected_label.value,
            vector=prediction["vector"],
            quality=float(prediction["quality"]),
            detection_confidence=float(prediction["detection_confidence"]),
            source_sha256=digest,
            source_kind="user_correction",
        )
    return {"corrected": True, "promoted": promoted, "references": db.counts()}


@app.delete("/api/v1/data", dependencies=[Depends(require_token)])
def erase_data() -> dict[str, bool]:
    db.erase_all()
    return {"erased": True}

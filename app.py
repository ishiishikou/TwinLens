import os
import secrets
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from flask import Flask, abort, jsonify, request, send_from_directory

from decision import LABEL_A, LABEL_B, OTHER, UNCERTAIN, decide

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("TWINLENS_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("TWINLENS_MODEL_DIR", ROOT / "models"))
DB_PATH = DATA_DIR / "twinlens.db"
TOKEN = os.getenv("TWINLENS_TOKEN", "")
KEY = os.getenv("TWINLENS_KEY", "")
ACCEPT = float(os.getenv("TWINLENS_ACCEPT_THRESHOLD", "0.48"))
MARGIN = float(os.getenv("TWINLENS_MARGIN_THRESHOLD", "0.06"))
OTHER_THRESHOLD = float(os.getenv("TWINLENS_OTHER_THRESHOLD", "0.25"))
MIN_QUALITY = float(os.getenv("TWINLENS_MIN_QUALITY", "25"))
MIN_DETECTION = float(os.getenv("TWINLENS_MIN_DETECTION", "0.80"))
MAX_PIXELS = int(os.getenv("TWINLENS_MAX_PIXELS", "40000000"))

if not TOKEN or not KEY:
    raise RuntimeError("TWINLENS_TOKEN and TWINLENS_KEY are required")
try:
    CIPHER = Fernet(KEY.encode())
except (ValueError, TypeError) as exc:
    raise RuntimeError("TWINLENS_KEY must be a Fernet key") from exc

DATA_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


class FaceEngine:
    def __init__(self):
        detector = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
        recognizer = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
        if not detector.exists() or not recognizer.exists():
            raise RuntimeError(f"Face models are missing in {MODEL_DIR}")
        self.detector = cv2.FaceDetectorYN_create(str(detector), "", (320, 320), MIN_DETECTION, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF_create(str(recognizer), "")

    def extract(self, raw):
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("画像を読み込めません")
        height, width = image.shape[:2]
        if height * width > MAX_PIXELS:
            raise ValueError("画像の画素数が上限を超えています")
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        results = []
        for face in [] if faces is None else faces:
            x, y, w, h = [int(v) for v in face[:4]]
            x, y = max(0, x), max(0, y)
            crop = image[y:min(height, y + h), x:min(width, x + w)]
            quality = 0.0 if crop.size == 0 or min(w, h) < 80 else float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            aligned = self.recognizer.alignCrop(image, face)
            vector = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                continue
            results.append({
                "box": [x, y, w, h],
                "detection": float(face[14]),
                "quality": quality,
                "vector": vector / norm,
            })
        return results


ENGINE = FaceEngine()


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                person TEXT NOT NULL CHECK(person IN ('A','B')),
                vector BLOB NOT NULL,
                source TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                predicted TEXT NOT NULL,
                score_a REAL NOT NULL,
                score_b REAL NOT NULL,
                quality REAL NOT NULL,
                corrected TEXT,
                created_at INTEGER NOT NULL
            );
        """)


def encrypt(vector):
    return CIPHER.encrypt(vector.astype(np.float32).tobytes())


def decrypt(blob):
    try:
        return np.frombuffer(CIPHER.decrypt(blob), dtype=np.float32)
    except InvalidToken as exc:
        raise RuntimeError("保存済み特徴量を復号できません。TWINLENS_KEYを確認してください") from exc


def references(person):
    with connect() as conn:
        rows = conn.execute("SELECT vector FROM embeddings WHERE person=?", (person,)).fetchall()
    return [decrypt(row["vector"]) for row in rows]


def similarity(vector, refs):
    if not refs:
        return -1.0
    # ponytail: O(n) scan; replace with FAISS only after reference vectors reach thousands.
    scores = sorted((float(np.dot(vector, ref)) for ref in refs), reverse=True)
    return float(np.mean(scores[:min(3, len(scores))]))


def require_token():
    supplied = request.headers.get("X-API-Token", "")
    if not secrets.compare_digest(supplied, TOKEN):
        abort(401)


def uploaded(name="image"):
    file = request.files.get(name)
    if not file or not file.filename:
        abort(400, "画像ファイルが必要です")
    return file.read()


@app.errorhandler(400)
@app.errorhandler(401)
@app.errorhandler(404)
@app.errorhandler(413)
def expected_error(error):
    return jsonify(error=getattr(error, "description", str(error))), error.code


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    require_token()
    with connect() as conn:
        counts = {row["person"]: row["count"] for row in conn.execute(
            "SELECT person, COUNT(*) AS count FROM embeddings GROUP BY person"
        )}
    return jsonify(A=counts.get("A", 0), B=counts.get("B", 0), thresholds={
        "accept": ACCEPT, "margin": MARGIN, "other": OTHER_THRESHOLD,
        "min_quality": MIN_QUALITY, "min_detection": MIN_DETECTION,
    })


@app.post("/api/enroll/<person>")
def enroll(person):
    require_token()
    person = person.upper()
    if person not in (LABEL_A, LABEL_B):
        abort(400, "personはAまたはBです")
    files = request.files.getlist("files")
    if not files:
        abort(400, "filesが必要です")
    accepted, rejected = 0, []
    with connect() as conn:
        for file in files:
            try:
                faces = ENGINE.extract(file.read())
                if len(faces) != 1:
                    raise ValueError("登録画像は顔が1つだけ必要です")
                face = faces[0]
                if face["detection"] < MIN_DETECTION or face["quality"] < MIN_QUALITY:
                    raise ValueError("顔の検出信頼度または画像品質が不足しています")
                conn.execute(
                    "INSERT INTO embeddings(person, vector, source, created_at) VALUES(?,?,?,?)",
                    (person, encrypt(face["vector"]), "enroll", int(time.time())),
                )
                accepted += 1
            except ValueError as exc:
                rejected.append({"file": file.filename, "reason": str(exc)})
    return jsonify(accepted=accepted, rejected=rejected)


@app.post("/api/identify")
def identify():
    require_token()
    try:
        faces = ENGINE.extract(uploaded())
    except ValueError as exc:
        abort(400, str(exc))
    if not faces:
        return jsonify(overall="other", faces=[])
    refs_a, refs_b = references(LABEL_A), references(LABEL_B)
    if not refs_a or not refs_b:
        abort(400, "双子Aと双子Bを先に登録してください")
    output = []
    with connect() as conn:
        for face in faces:
            score_a = similarity(face["vector"], refs_a)
            score_b = similarity(face["vector"], refs_b)
            label = decide(score_a, score_b, face["quality"], face["detection"],
                           accept=ACCEPT, margin=MARGIN, other=OTHER_THRESHOLD,
                           min_quality=MIN_QUALITY, min_detection=MIN_DETECTION)
            prediction_id = secrets.token_urlsafe(12)
            conn.execute("""
                INSERT INTO predictions(id, vector, predicted, score_a, score_b, quality, created_at)
                VALUES(?,?,?,?,?,?,?)
            """, (prediction_id, encrypt(face["vector"]), label, score_a, score_b,
                  face["quality"], int(time.time())))
            output.append({
                "id": prediction_id,
                "label": label,
                "box": face["box"],
                "score_a": round(score_a, 4),
                "score_b": round(score_b, 4),
                "quality": round(face["quality"], 1),
                "detection": round(face["detection"], 4),
            })
    return jsonify(overall="faces", faces=output)


@app.post("/api/corrections/<prediction_id>")
def correct(prediction_id):
    require_token()
    body = request.get_json(silent=True) or {}
    label = body.get("label")
    if label not in (LABEL_A, LABEL_B, UNCERTAIN, OTHER):
        abort(400, "labelが不正です")
    add_reference = bool(body.get("add_reference"))
    with connect() as conn:
        row = conn.execute("SELECT vector FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        if not row:
            abort(404)
        conn.execute("UPDATE predictions SET corrected=? WHERE id=?", (label, prediction_id))
        if add_reference:
            if label not in (LABEL_A, LABEL_B):
                abort(400, "A/B以外は参照データに追加できません")
            conn.execute(
                "INSERT INTO embeddings(person, vector, source, created_at) VALUES(?,?,?,?)",
                (label, row["vector"], "correction", int(time.time())),
            )
    return jsonify(ok=True)


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=False)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TWINLENS_MODEL_DIR=/models TWINLENS_DATA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends curl libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir /models /data \
 && curl -LfsS -o /models/face_detection_yunet_2023mar.onnx https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx \
 && curl -LfsS -o /models/face_recognition_sface_2021dec.onnx https://huggingface.co/opencv/face_recognition_sface/resolve/c140188d35b7d0050f2dcfdfb8fe3e98d516744f/face_recognition_sface_2021dec.onnx \
 && test "$(stat -c%s /models/face_detection_yunet_2023mar.onnx)" -gt 100000 \
 && test "$(stat -c%s /models/face_recognition_sface_2021dec.onnx)" -gt 1000000
COPY app.py decision.py ./
COPY static ./static
RUN useradd -u 10001 -r -s /usr/sbin/nologin twinlens && chown -R twinlens /data /app /models
USER twinlens
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)" || exit 1
CMD ["python", "app.py"]

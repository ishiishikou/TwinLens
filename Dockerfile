FROM python:3.12-slim AS runtime

ARG OPENCV_ZOO_REV=47534e27c9851bb1128ccc0102f1145e27f23f98
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TWINLENS_MODEL_DIR=/models \
    TWINLENS_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN mkdir -p /models /data \
    && curl -L --fail --retry 3 -o /models/face_detection_yunet_2023mar.onnx \
       "https://github.com/opencv/opencv_zoo/raw/${OPENCV_ZOO_REV}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    && curl -L --fail --retry 3 -o /models/face_recognition_sface_2021dec.onnx \
       "https://github.com/opencv/opencv_zoo/raw/${OPENCV_ZOO_REV}/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
    && echo "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  /models/face_detection_yunet_2023mar.onnx" | sha256sum -c - \
    && echo "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  /models/face_recognition_sface_2021dec.onnx" | sha256sum -c -

COPY app ./app
RUN useradd --create-home --uid 10001 twinlens && chown -R twinlens:twinlens /app /data /models
USER twinlens

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

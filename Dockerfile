# Inference service image (deployed to Kubernetes).
# Build after training has produced models/model.joblib:
#   docker build -t ml-inference:1.0.0 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.115.* "uvicorn[standard]==0.32.*" \
    scikit-learn joblib numpy

COPY inference/ ./inference/
COPY models/model.joblib ./models/model.joblib

RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]

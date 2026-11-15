# Inference service image (deployed to Kubernetes).
# Build after training has produced models/model.joblib:
#   docker build -t ml-inference:1.0.0 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Library versions are pinned to the training image (Dockerfile.airflow / apache/airflow:3.0.6) so the
# pickled model bundle loads identically; `dill` is needed because that image pickles numpy scalars with it.
RUN pip install --no-cache-dir fastapi==0.115.* "uvicorn[standard]==0.32.*"     scikit-learn==1.7.1 numpy==1.26.4 joblib==1.5.1 dill==0.3.1.1

COPY inference/ ./inference/
COPY models/model.joblib ./models/model.joblib

RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]

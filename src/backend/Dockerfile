FROM python:3.12-slim

ARG BACKEND_PORT
ARG MLFLOW_TRACKING_URI

ARG MODEL_URI_PROD
ARG MODEL_URI_DEV
ARG MODEL_URI_BACKUP

ARG WEAKLINK_MODEL_URI_PROD
ARG WEAKLINK_MODEL_URI_DEV
ARG WEAKLINK_MODEL_URI_BACKUP

WORKDIR /backend_app

ENV BACKEND_PORT=${BACKEND_PORT} \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    MODEL_URI_PROD=${MODEL_URI_PROD} \
    MODEL_URI_DEV=${MODEL_URI_DEV} \
    MODEL_URI_BACKUP=${MODEL_URI_BACKUP} \
    WEAKLINK_MODEL_URI_PROD=${WEAKLINK_MODEL_URI_PROD} \
    WEAKLINK_MODEL_URI_DEV=${WEAKLINK_MODEL_URI_DEV} \
    WEAKLINK_MODEL_URI_BACKUP=${WEAKLINK_MODEL_URI_BACKUP} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/backend_app/.venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv /backend_app/.venv

COPY requirements.txt .
RUN pip install --upgrade pip~=25.3 && \
    pip install --no-cache-dir -r requirements.txt

RUN useradd -m appuser && chown -R appuser /backend_app /backend_app/.venv

USER appuser

COPY --chown=appuser:appuser . .

EXPOSE ${BACKEND_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${BACKEND_PORT}/docs || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${BACKEND_PORT}"]
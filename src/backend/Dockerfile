FROM python:3.12-slim

ARG BACKEND_PORT

WORKDIR /backend_app

ENV BACKEND_PORT=${BACKEND_PORT:-8080} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/.venv/bin:$PATH"

RUN python -m venv /opt/.venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser /backend_app
USER appuser

EXPOSE ${BACKEND_PORT}
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${BACKEND_PORT}"]
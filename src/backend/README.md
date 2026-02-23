# Backend API

FastAPI backend for the 4dt907 ML data-intensive system. Serves ML model predictions via REST API.

## Getting Started

### Prerequisites

- Python 3.12.x
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
cd src/backend

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|----------|-------------|
| `BACKEND_PORT` | Port for the backend (default: `8080`) |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI |
| `MODEL_URI_PROD` | Champion model URI (e.g. `models:/YourModel@prod`) |
| `MODEL_URI_DEV` | Dev/latest model URI |
| `WEAKLINK_MODEL_URI_PROD` | Weakest-link champion model URI |
| `WEAKLINK_MODEL_URI_DEV` | Weakest-link dev model URI |

### Running

```bash
# From src/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

API available at:

- API: <http://localhost:8080>
- Interactive docs: <http://localhost:8080/docs>

### Tests

```bash
pytest
pytest -v          # verbose
pytest --cov=app   # with coverage
```

### Linting

```bash
flake8 .
black .
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root – lists available routes |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/predict/champion` | Predict with champion model |
| `POST` | `/api/v1/predict/latest` | Predict with latest model |
| `POST` | `/api/v1/weakest-link/champion` | Weakest-link champion prediction |
| `POST` | `/api/v1/weakest-link/latest` | Weakest-link latest prediction |
| `GET` | `/api/v1/model-info/champion` | Champion model metadata |
| `GET` | `/api/v1/model-info/latest` | Latest model metadata |

## Docker

```bash
# From src/backend
docker build -t 4dt907-backend .
docker run -p 8080:8080 4dt907-backend
```

Or use docker compose from `src/`:

```bash
docker compose up -d backend
```

## Project Structure

```text
backend/
├── app/
│   ├── main.py          # Application entry point
│   ├── api/             # Route handlers (v1, v2, health)
│   ├── schemas/         # Pydantic request/response models
│   └── services/        # ML model loading and prediction logic
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```


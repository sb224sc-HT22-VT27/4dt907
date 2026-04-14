# Backend API

FastAPI backend service for 4dt907 ML data-intensive system.

## Features

- **Squat classification** — `POST /api/v1/squat/classify` accepts MediaPipe 3-D keypoints,
  calculates knee angles (law of cosines) and returns `Deep` / `Shallow` / `Invalid` with a
  confidence score. Uses a PyTorch model when available; falls back to rule-based thresholds.
- **Expert-score prediction** — `POST /api/v1/predict/champion` and `/latest` (MLflow model)
- **Weakest-link classification** — `POST /api/v1/weakest-link/classify` (MLflow model)

## Getting Started

### Prerequisites

- [Python 3.12.10](https://www.python.org/downloads/release/python-31210/)
- For vercel local testing use [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For docker local testing use pip

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For vercel local testing
uv sync
```

### Environment variables (.env)

Create `.env` files using `.env.example` as a template:

`src/.env` (For docker)
`.env` (in root for `vercel dev`)

### Running the Application

```bash
# Development server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Or using Python directly
python -m app.main
```

The API available at:

- API: <http://localhost:8080>
- Interactive docs: <http://localhost:8080/docs>

### Running Tests

```bash
pytest -v           # Verbose output
pytest --cov=app    # With coverage report
```

### Linting

```bash
flake8 .
black .
```

## API Endpoints

### Root Endpoints

- `GET /health` — liveness probe
- `GET /` — application info

### API v1

#### Squat Analysis

- `POST /api/v1/squat/classify` — classify squat depth from MediaPipe 3-D keypoints

  **Request** (`application/json`):
  ```json
  {
    "keypoints_3d": [
      { "name": "left_hip",   "x": 0.10, "y": 0.01, "z": -0.01 },
      { "name": "left_knee",  "x": 0.18, "y": 0.37, "z": -0.03 },
      { "name": "left_ankle", "x": 0.15, "y": 0.70, "z":  0.12 }
    ]
  }
  ```
  At minimum `left_hip`, `left_knee`, `left_ankle`, `right_hip`, `right_knee`, `right_ankle`
  must be present.

  **Response**:
  ```json
  {
    "classification": "Deep",
    "left_knee_angle": 82.3,
    "right_knee_angle": 84.1,
    "confidence": 0.91
  }
  ```

#### Prediction

- `POST /api/v1/predict/champion`
- `POST /api/v1/predict/latest`

## Docker

Use the docker compose file to build entire project which uses the local Dockerfile for the backend.

## MLflow Integration

The `/api/v1/predict/*` and `/api/v1/weakest-link/*` endpoints load models from MLflow
(DagsHub) at startup. Set `MLFLOW_TRACKING_URI` and the appropriate `MODEL_URI_*` / 
`WEAKLINK_MODEL_URI_*` environment variables (see `.env.example`).

## PyTorch Squat Model (optional)

The squat classifier will use a trained PyTorch checkpoint if present at:

```
src/backend/app/models/squat_model.pt
```

The model is loaded **once at startup** and cached. If the file is missing or `torch` is not
installed the endpoint automatically falls back to a rule-based angle-threshold classifier so
the endpoint remains functional without a trained model.

## Project Structure (Update as needed)

```text
backend/
├── app/
│   ├── main.py          # Application entry point
│   ├── api/             # API route handlers
│   ├── schemas/         # Schemas
│   └── services/        # Business logic
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

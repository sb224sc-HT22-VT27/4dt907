# Backend API

FastAPI backend service for 4dt907 ML data-intensive system.

## Features

- **Squat classification** — `POST /api/v1/squat/classify` accepts MediaPipe 3-D keypoints,
  calculates knee angles (law of cosines) and returns `Deep` / `Shallow` / `Invalid` with a
  confidence score. Uses the MLflow-hosted z-predictor model to reconstruct z from x/y.
- **Expert-score prediction** — `POST /api/v1/predict/champion` and `/latest` (MLflow model)
- **Weakest-link classification** — `POST /api/v1/weakest-link/champion` and `/latest`
- **Z-predictor** — `POST /api/v1/z-predictor/champion` and `/latest`

## Getting Started

### Prerequisites

- [Python 3.9.12](https://www.python.org/downloads/release/python-3912/)
- For docker local testing use pip

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment variables (.env)

Create `.env` files using `.env.example` as a template:

`src/.env` (For docker)
`.env`

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

#### Z-Predictor

Predicts the z-axis (depth) value for a set of MediaPipe 2-D keypoints using an MLflow-hosted
regression model. The squat classifier calls this internally to improve depth estimation.

- `POST /api/v1/z-predictor/champion` — production z-predictor model
- `POST /api/v1/z-predictor/latest` — development z-predictor model

  **Request** — same schema as `/api/v1/predict/*` (a flat feature vector).

  **Response**:

  ```json
  { "prediction": 0.042, "model_uri": "models:/…", "run_id": "abc123" }
  ```

  Requires `Z_MODEL_URI_PROD` (champion) or `Z_MODEL_URI_DEV` (latest) in the environment.

## Docker

Use the docker compose file to build entire project which uses the local Dockerfile for the backend.

## MLflow Integration

The `/api/v1/predict/*` and `/api/v1/weakest-link/*` endpoints load models from MLflow
(DagsHub) at startup. Set `MLFLOW_TRACKING_URI` and the appropriate `MODEL_URI_*` /
`WEAKLINK_MODEL_URI_*` / `Z_MODEL_URI_*` environment variables (see `.env.example`).

## Project Structure

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

# Backend API

FastAPI backend service for 4dt907 ML data-intensive system.

## Features

## Getting Started

### Prerequisites

- Python 3.12.10
- For vercel local testing use [uv](https://docs.astral.sh/uv/)
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

### API v1

#### Prediction

- `POST /api/v1/predict/champion`
- `POST /api/v1/predict/latest`

## Docker

Use the docker compose file to build entire project which uses the local Dockerfile for the backend.

## MLflow Integration

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

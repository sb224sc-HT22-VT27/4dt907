# Backend API

FastAPI backend service for 4dt907 ML data-intensive system.

## Features

- RESTful API with FastAPI
- Auto-generated OpenAPI documentation
- MLflow integration for ML experiment tracking
- CORS enabled for frontend integration
- Health check endpoint
- Pytest test suite

## Getting Started

### Prerequisites

- Python 3.12.x
- pip

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Development server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m app.main
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=app tests/
```

### Linting

```bash
# Check code style
flake8 .

# Format code with black
black .
```

## API Endpoints

### Root Endpoints

- `GET /` - Hello World message
- `GET /health` - Health check

### API v1

- `GET /api/v1/hello/{name}` - Personalized greeting

## Docker

```bash
# Build image
docker build -t 4dt907-backend .

# Run container
docker run -p 8000:8000 4dt907-backend
```

## MLflow Integration

The backend is configured to connect to MLflow tracking server:

```python
import mlflow

# MLflow tracking URI is set via environment variable
# MLFLOW_TRACKING_URI=http://mlflow:5000
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # Application entry point
│   ├── api/             # API route handlers
│   ├── models/          # Data models
│   └── services/        # Business logic
├── tests/
│   ├── __init__.py
│   └── test_main.py     # Test cases
├── Dockerfile
├── requirements.txt
└── README.md
```

## Environment Variables

- `MLFLOW_TRACKING_URI` - MLflow tracking server URL (default: http://mlflow:5000)

## Contributing

Follow the [contribution guidelines](../../CONTRIBUTING.md) in the root repository.

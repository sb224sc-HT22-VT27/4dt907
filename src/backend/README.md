# Backend API

FastAPI backend service for 4dt907 ML data-intensive system.

## Features

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

### Environment variables (.env)

This backend loads MLflow models from DagsHub. Create a `.env` file for local development:

`src/backend/.env`


### Running the Application

```bash
# Development server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m app.main
```

The API will be available at:

- API: <http://localhost:8000>
- Interactive docs: <http://localhost:8000/docs>
- Alternative docs: <http://localhost:8000/redoc>

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

### API v1

#### Prediction

- `POST /api/v1/predict/champion`
- `POST /api/v1/predict/latest`

## Docker

```bash
# Build image
docker build -t 4dt907-backend .

# Run container
docker run -p 8000:8000 4dt907-backend
```

## MLflow Integration

## Project Structure (Update as needed)

```text
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

## Contributing

Follow the [contribution guidelines](../../CONTRIBUTING.md) in the root repository.

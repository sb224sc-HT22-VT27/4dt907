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

### Environment Variables (.env)

This backend loads MLflow models from DagsHub. Create a `.env` file for local development:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and configure the following variables:

```bash
# Required: MLflow tracking URI (DagsHub)
MLFLOW_TRACKING_URI=https://dagshub.com/<YOUR_USERNAME>/<YOUR_REPO>.mlflow

# Required: Model URIs pointing to your trained models
MODEL_URI_PROD=models:/Best_Production_Model@production
MODEL_URI_DEV=models:/Latest_Model@latest

# Optional: Backup model
MODEL_URI_BACKUP=models:/Backup_Model@backup

# Optional: Model registry names (fallback if direct URIs not set)
MLFLOW_BEST_MODEL_NAME=Best_Production_Model
MLFLOW_LATEST_MODEL_NAME=Latest_Model

# Optional: Authentication for private DagsHub repositories
# MLFLOW_TRACKING_USERNAME=<YOUR_DAGSHUB_USERNAME>
# MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>
```

See the [Deployment Guide](../../DEPLOYMENT.md) for detailed configuration instructions.


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

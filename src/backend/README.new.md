# Backend API

Express.js backend service for 4dt907 ML data-intensive system.

## Features

- RESTful API with Express.js
- MLflow model integration via REST API
- Request validation with Joi
- Comprehensive test coverage with Jest
- Docker containerization
- CORS support for frontend integration

## Getting Started

### Prerequisites

- Node.js 22.x (LTS)
- npm

### Installation

```bash
# Install dependencies
npm install
```

### Environment variables (.env)

This backend loads MLflow models from DagsHub. Create a `.env` file for local development:

`src/.env`

```env
# Web service
BACKEND_PORT=8080
BACKEND_URL="http://backend:${BACKEND_PORT}"
FRONTEND_PORT=3030

# MLflow / DagsHub
MLFLOW_TRACKING_URI="https://dagshub.com/<Repo-owner>/<Repo-name>.mlflow"

# Model URIs
MODEL_URI_PROD="models:/<model>@<tag>"
MODEL_URI_DEV="models:/<model>@<tag>"
MODEL_URI_BACKUP="models:/<model>@<tag>"

# Weakest-link model URIs
WEAKLINK_MODEL_URI_PROD="models:/<model>@<tag>"
WEAKLINK_MODEL_URI_DEV="models:/<model>@<tag>"
WEAKLINK_MODEL_URI_BACKUP="models:/<model>@<tag>"
```

### Running the Application

```bash
# Development server with hot reload
npm run dev

# Production server
npm start
```

The API will be available at:

- API: <http://localhost:8080>
- Health check: <http://localhost:8080/health>
- API endpoints: See API Endpoints section below

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Linting

```bash
# Check code style
npm run lint
```

## API Endpoints

### Root Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check endpoint

### API v1

#### Prediction Endpoints

- `POST /api/v1/predict/champion` - Predictions using champion model
  - Request body: `{"features": [float, ...]}`
  - Response: `{"prediction": float, "model_uri": string}`

- `POST /api/v1/predict/latest` - Predictions using latest model
  - Request body: `{"features": [float, ...]}`
  - Response: `{"prediction": float, "model_uri": string}`

#### Model Information Endpoints

- `GET /api/v1/model-info/champion` - Get champion model metadata
- `GET /api/v1/model-info/latest` - Get latest model metadata

#### Weakest-Link Endpoints

- `POST /api/v1/weakest-link/champion` - Weakest-link predictions (champion)
  - Request body: `{"features": [float, ...]}`
  - Response: `{"prediction": string, "model_uri": string}`

- `POST /api/v1/weakest-link/latest` - Weakest-link predictions (latest)
  - Request body: `{"features": [float, ...]}`
  - Response: `{"prediction": string, "model_uri": string}`

- `GET /api/v1/model-info/weakest-link/champion` - Weakest-link champion model metadata
- `GET /api/v1/model-info/weakest-link/latest` - Weakest-link latest model metadata

### API v2

- `GET /api/v2/status` - API v2 status endpoint

## Docker

```bash
# Build image
docker build -f Dockerfile.new -t 4dt907-backend .

# Run container
docker run -p 8080:8080 4dt907-backend
```

## MLflow Integration

This Express.js backend integrates with MLflow using the REST API instead of the Python SDK. For production use:

1. **MLflow Model Serving**: Deploy models using MLflow's model serving capabilities
2. **Python Microservice**: Use a separate Python service for ML operations
3. **REST API**: Call MLflow REST API endpoints for model registry and metadata

The current implementation provides:
- Model registry integration via MLflow REST API
- Alias resolution (prod/dev/backup variants)
- Model caching for performance
- Feature validation

**Note**: The prediction endpoints return placeholder values. For production, integrate with:
- MLflow Model Serving deployment
- A Python microservice handling actual predictions
- Deployed model endpoints

## Project Structure

```text
backend/
├── src/
│   ├── index.js                 # Application entry point
│   ├── api/
│   │   ├── health.js            # Health check endpoint
│   │   ├── v1/
│   │   │   ├── router.js        # v1 API router
│   │   │   └── endpoints/       # v1 endpoint handlers
│   │   │       ├── predict.js   # Prediction endpoints
│   │   │       ├── modelInfo.js # Model info endpoints
│   │   │       └── weakestLink.js # Weakest-link endpoints
│   │   └── v2/
│   │       └── router.js        # v2 API router
│   ├── services/
│   │   ├── modelService.js           # MLflow model service
│   │   └── weaklinkModelService.js   # Weakest-link model service
│   └── schemas/
│       └── prediction.js        # Request/response schemas
├── tests/
│   └── index.test.js            # Test cases
├── Dockerfile.new               # Node.js Dockerfile
├── package.json
└── README.md
```

## Contributing

Follow the [contribution guidelines](../../CONTRIBUTING.md) in the root repository.

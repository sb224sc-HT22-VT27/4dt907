# Model Deployment & API Service Guide

This guide explains how to deploy the champion ML model as a web service and connect the frontend, backend, and DagsHub models.

## Architecture Overview

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Frontend   │─────>│   Backend    │─────>│   DagsHub    │
│  (React +    │      │  (FastAPI +  │      │   MLflow     │
│   Nginx)     │      │   MLflow)    │      │   Registry   │
└──────────────┘      └──────────────┘      └──────────────┘
     Port 3030            Port 8080         Remote MLflow URI
```

The system consists of three main components:
1. **Frontend**: React application that provides a web UI for making predictions
2. **Backend**: FastAPI service that loads models from DagsHub and serves predictions
3. **DagsHub**: Remote MLflow model registry hosting trained models

## Prerequisites

- Docker and Docker Compose installed
- Python 3.12.x (for local development)
- Node.js 22.x (for frontend development)
- Access to a DagsHub repository with trained MLflow models
- DagsHub account and token (if using private repositories)

## Configuration

### Step 1: Set Up Environment Variables

The service requires configuration through environment variables to connect to your DagsHub MLflow registry.

#### For Docker Compose Deployment

Create a `.env` file in the `src/` directory:

```bash
cd src
cp .env.example .env
```

Edit `.env` and set your actual values:

```bash
# Backend and Frontend Ports
BACKEND_PORT=8080
FRONTEND_PORT=3030

# MLflow / DagsHub Configuration
MLFLOW_TRACKING_URI=https://dagshub.com/<YOUR_USERNAME>/<YOUR_REPO>.mlflow

# Model URIs - Point to your trained models in DagsHub
MODEL_URI_PROD=models:/Best_Production_Model@production
MODEL_URI_DEV=models:/Latest_Model@latest

# DagsHub Authentication (if using private repository)
MLFLOW_TRACKING_USERNAME=<YOUR_DAGSHUB_USERNAME>
MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>
```

#### For Local Backend Development

Create a `.env` file in the `src/backend/` directory:

```bash
cd src/backend
cp .env.example .env
```

Edit the file with the same MLflow configuration as above.

### Step 2: Understanding Model URIs

The backend supports multiple ways to reference models:

#### Option A: Direct Model URIs (Recommended)

Use environment variables to specify exact model locations:
- `MODEL_URI_PROD`: Champion/production model
- `MODEL_URI_DEV`: Latest/development model
- `MODEL_URI_BACKUP`: Backup model

Supported URI formats:
```
models:/ModelName@alias          # Using registry alias
models:/ModelName/1              # Specific version number
runs:/<run-id>/model            # Direct run artifact
```

#### Option B: Registry Model Names (Fallback)

If direct URIs are not set, the service falls back to looking up models by name:
- `MLFLOW_BEST_MODEL_NAME`: Name of champion model in registry
- `MLFLOW_LATEST_MODEL_NAME`: Name of latest model in registry

### Step 3: Getting Your Configuration from DagsHub

1. **Log in to DagsHub**: Go to https://dagshub.com and sign in
2. **Navigate to your repository**: Find the repo with your ML models
3. **Access MLflow**: Click on "Experiments" or "MLflow" tab
4. **Copy the tracking URI**: 
   - It will look like: `https://dagshub.com/<username>/<repo>.mlflow`
5. **Find your models**:
   - Go to the "Models" tab in MLflow
   - Note the model names and versions you want to use
6. **Get authentication token**:
   - Go to Settings > Tokens in DagsHub
   - Create a new token or use an existing one
   - Use your DagsHub username and token as credentials

## Deployment Options

### Option 1: Docker Compose (Full Stack - Recommended)

This runs both frontend and backend in containers.

```bash
# From the src/ directory
cd src

# Make sure .env is configured
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

Access the application:
- **Frontend**: http://localhost:3030
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

### Option 2: Local Development (Backend Only)

Run the backend locally for development and testing.

```bash
cd src/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure .env is configured

# Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Access the API:
- **API**: http://localhost:8080
- **Interactive Docs**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

### Option 3: Production Deployment

For production deployment, you can:

1. **Use Docker Compose with production settings**
2. **Deploy to cloud platforms** (AWS, GCP, Azure):
   - Push Docker images to container registry
   - Deploy using Kubernetes, ECS, Cloud Run, etc.
3. **Use serverless** (AWS Lambda, Google Cloud Functions)
4. **Deploy frontend separately** (Netlify, Vercel, S3 + CloudFront)

## Using the API

### API Endpoints

#### Health Check
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok"
}
```

#### Predict with Champion Model
```bash
curl -X POST http://localhost:8080/api/v1/predict/champion \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

#### Predict with Latest Model
```bash
curl -X POST http://localhost:8080/api/v1/predict/latest \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

Response format:
```json
{
  "prediction": 0.8543,
  "model_uri": "models:/Best_Production_Model/3"
}
```

### Using the Web Interface

1. Open http://localhost:3030 in your browser
2. Select a model (Champion or Latest)
3. Enter comma-separated feature values (e.g., `1, 2, 3`)
4. Click "Predict"
5. View the prediction result and model URI

## Testing

### Backend Tests

```bash
cd src/backend

# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/api/v1/endpoints/test_predict.py -v
```

### Test Without MLflow Connection

The tests use mocks and don't require an actual MLflow connection. This allows you to test the API logic independently.

### Manual Testing

Use the interactive API docs at http://localhost:8080/docs to:
1. View all available endpoints
2. Try out the API with sample data
3. See request/response schemas
4. Test error handling

## Troubleshooting

### "MLFLOW_TRACKING_URI is not set"

**Solution**: Make sure your `.env` file is in the correct location (`src/.env` or `src/backend/.env`) and contains the `MLFLOW_TRACKING_URI` variable.

### "No versions found for model"

**Solution**: 
1. Verify the model name is correct in your `.env` file
2. Check that models are registered in your DagsHub MLflow registry
3. Ensure you have access to the DagsHub repository

### "INVALID_PARAMETER_VALUE" with alias URIs

The backend automatically handles this by falling back to version-based URIs. However, you can:
1. Use version numbers instead of aliases: `models:/ModelName/1`
2. Check that your model versions are properly registered in DagsHub

### Authentication Errors

**Solution**:
1. Verify your DagsHub username and token are correct
2. Ensure the token has appropriate permissions
3. For public repositories, authentication may not be needed
4. Check if you need to uncomment the auth variables in `.env`

### Connection Refused / Network Errors

**Solution**:
1. Verify Docker containers are running: `docker compose ps`
2. Check logs: `docker compose logs backend`
3. Ensure ports 3030 and 8080 are not in use by other applications
4. Verify network connectivity to dagshub.com

### Model Expects Different Number of Features

**Solution**: The model expects a specific number of input features. Check:
1. The model's training data feature count
2. Send the correct number of features in your request
3. Use the `/api/v1/model-info/latest` endpoint to check expected features

### Docker Volume Issues

**Solution**:
```bash
# Stop and remove containers with volumes
docker compose down -v

# Rebuild and restart
docker compose up -d --build
```

## Model Management

### Updating Models

When you train and register a new model in DagsHub:

1. **Using aliases** (Recommended):
   - Models with aliases update automatically
   - No code changes needed
   - Just update the alias in DagsHub to point to the new version

2. **Using version numbers**:
   - Update the `MODEL_URI_*` variables in `.env`
   - Restart the backend service

3. **Clear model cache**:
   ```bash
   # Restart backend to clear cache
   docker compose restart backend
   ```

### Monitoring Model Performance

1. Check predictions in the frontend UI
2. Review logs: `docker compose logs backend`
3. Use MLflow in DagsHub to track model metrics
4. Set up monitoring for prediction latency and accuracy

## Security Considerations

1. **Never commit `.env` files** - They contain sensitive credentials
2. **Use environment-specific credentials** - Different tokens for dev/prod
3. **Rotate tokens regularly** - Update DagsHub tokens periodically
4. **Use HTTPS in production** - Deploy behind SSL/TLS
5. **Implement rate limiting** - Protect the API from abuse
6. **Add authentication** - For production use, add API keys or OAuth

## Advanced Configuration

### Custom Model Loading

The backend supports custom model loading logic in `app/services/model_service.py`. You can:
- Add caching strategies
- Implement A/B testing
- Add model version pinning
- Implement gradual rollouts

### CORS Configuration

CORS is configured in `app/main.py`. Modify `ALLOWED_ORIGINS` for your domain:

```python
ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://www.yourdomain.com",
]
```

### Scaling

For high traffic:
1. Use container orchestration (Kubernetes)
2. Add load balancing
3. Implement model serving frameworks (TensorFlow Serving, TorchServe)
4. Cache predictions for common inputs
5. Use async workers for prediction processing

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review backend logs: `docker compose logs backend`
3. Check DagsHub MLflow registry status
4. Consult FastAPI docs: https://fastapi.tiangolo.com
5. Consult MLflow docs: https://www.mlflow.org

## Next Steps

1. ✅ Set up environment variables
2. ✅ Deploy using Docker Compose
3. ✅ Test predictions via web interface
4. ✅ Test API endpoints with curl or API docs
5. 🔄 Train and register models in DagsHub
6. 🔄 Monitor model performance
7. 🔄 Plan production deployment

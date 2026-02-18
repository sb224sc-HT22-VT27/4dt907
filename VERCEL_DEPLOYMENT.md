# Vercel Deployment Guide

This guide explains how to deploy the 4dt907 project to Vercel as separate frontend and backend projects.

## Architecture

The project is split into two separate Vercel projects:

1. **Backend API** (`src/backend/`) - FastAPI application deployed as serverless functions
2. **Frontend** (`src/frontend/`) - React application deployed as a static site

## Prerequisites

- Vercel CLI installed: `npm i -g vercel`
- Vercel account and appropriate access to the organization

## Backend Deployment

### Initial Setup

1. Navigate to the backend directory:
   ```bash
   cd src/backend
   ```

2. Login to Vercel (if not already logged in):
   ```bash
   vercel login
   ```

3. Link to Vercel project (first time only):
   ```bash
   vercel link
   ```
   - Select or create a project for the backend (e.g., `4dt907-backend`)

4. Set environment variables in Vercel dashboard or via CLI:
   ```bash
   vercel env add MLFLOW_TRACKING_URI
   vercel env add MODEL_URI_PROD
   vercel env add MODEL_URI_DEV
   vercel env add MODEL_URI_BACKUP
   # Add any other required environment variables
   ```

### Deploy Backend

```bash
# Deploy to preview
cd src/backend
vercel

# Deploy to production
vercel --prod
```

The backend will be available at: `https://your-backend-project.vercel.app`

### Backend Configuration

The `src/backend/vercel.json` configures:
- Python runtime for FastAPI
- Serverless function routing through `api/index.py`
- All routes proxied to the FastAPI application

## Frontend Deployment

### Initial Setup

1. Navigate to the frontend directory:
   ```bash
   cd src/frontend
   ```

2. Link to Vercel project (first time only):
   ```bash
   vercel link
   ```
   - Select or create a project for the frontend (e.g., `4dt907-frontend`)

3. Set the backend URL environment variable:
   ```bash
   # For production environment
   vercel env add BACKEND_URL production
   
   # For preview environment
   vercel env add BACKEND_URL preview
   
   # For development environment  
   vercel env add BACKEND_URL development
   ```
   - Set this to your backend deployment URL (e.g., `https://your-backend-project.vercel.app`)
   - The frontend build process will use this URL for API calls

### Deploy Frontend

```bash
# Deploy to preview
cd src/frontend
vercel

# Deploy to production
vercel --prod
```

The frontend will be available at: `https://your-frontend-project.vercel.app`

### Frontend Configuration

The `src/frontend/vercel.json` configures:
- Static build using Vite
- Output directory set to `dist`

The frontend uses the `BACKEND_URL` environment variable during build time. This should be set in the Vercel project settings or via the CLI. The Vite configuration (`vite.config.js`) already supports this variable for proxy configuration during development and build.

## Environment Variables

### Backend Environment Variables

Required variables for the backend:
- `MLFLOW_TRACKING_URI` - MLflow tracking server URL
- `MODEL_URI_PROD` - Production model URI
- `MODEL_URI_DEV` - Development model URI  
- `MODEL_URI_BACKUP` - Backup model URI
- Any DagsHub credentials if needed

### Frontend Environment Variables

Required variables for the frontend:
- `BACKEND_URL` - Backend API URL (e.g., `https://your-backend-project.vercel.app`)

## Local Development

For local development, continue using the existing setup:

```bash
# Backend (from src/backend)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Frontend (from src/frontend)
npm run dev
```

Or use Docker Compose:

```bash
cd src
docker compose up
```

## CI/CD Integration

To automate deployments, you can:

1. Add Vercel tokens and project IDs to GitHub Secrets:
   - `VERCEL_TOKEN` - Vercel authentication token
   - `VERCEL_ORG_ID` - Vercel organization ID
   - `VERCEL_BACKEND_PROJECT_ID` - Backend project ID
   - `VERCEL_FRONTEND_PROJECT_ID` - Frontend project ID
   - `BACKEND_URL` - Backend deployment URL for frontend to use

2. The GitHub Actions workflow (`.github/workflows/vercel-cd.yml`) will automatically deploy both projects on push to main

The workflow:
- Deploys backend first
- Then deploys frontend with backend URL configured
- Provides deployment summary with both URLs

Example workflow is already configured in `.github/workflows/vercel-cd.yml`.

## Troubleshooting

### Backend Issues

- **Import errors**: Ensure `api/index.py` correctly sets up Python paths
- **Environment variables**: Verify all required env vars are set in Vercel dashboard
- **Cold starts**: Serverless functions may have cold start delays

### Frontend Issues

- **API calls failing**: Verify `BACKEND_URL` is correctly set and points to deployed backend
- **CORS errors**: Backend should allow frontend origin in CORS middleware
- **Build errors**: Check Node.js version matches local development (22.x)

### General Tips

- Use `vercel logs` to view deployment logs
- Use `vercel env ls` to list environment variables
- Test preview deployments before promoting to production
- Keep backend and frontend URLs in sync

## Migration from Monorepo Deployment

The previous single `vercel.json` at root has been replaced with separate configurations:
- `src/backend/vercel.json` - Backend configuration
- `src/frontend/vercel.json` - Frontend configuration

This provides:
- Independent deployment cycles
- Better separation of concerns
- Easier scaling and configuration per service
- Clearer dependency management

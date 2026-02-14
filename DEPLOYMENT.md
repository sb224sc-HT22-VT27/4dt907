# Vercel Deployment Guide

This guide explains how to deploy the 4dt907 project to Vercel.

## Important: Deployment Size Considerations

⚠️ **The backend includes ML dependencies (mlflow, numpy, pandas) that can approach or exceed Vercel's free tier serverless function limit (50MB compressed).**

**Deployment Options:**

1. **Vercel Pro Plan** (Recommended for full-stack)
   - 250MB function size limit
   - Suitable for ML applications
   - ~$20/month per team member

2. **Frontend-Only on Vercel + External Backend** (Recommended for Free Tier)
   - Deploy frontend to Vercel (always free)
   - Host backend separately on Railway, Render, Fly.io, or Docker hosting
   - Configure `BACKEND_URL` in Vercel environment variables
   - Most cost-effective for production

3. **Full Stack on Vercel Free** (May fail)
   - Only if you optimize dependencies significantly
   - Risk of deployment failures
   - Not recommended for ML applications

**This guide covers all three options.**

---

## Overview

The project is configured as a monorepo with:
- **Frontend**: React + Vite application in `src/frontend/`
- **Backend**: FastAPI application wrapped as serverless functions in `api/`

## Architecture

### How It Works

1. **Frontend Deployment**
   - Vercel builds the React app using Vite
   - Static files are served from `src/frontend/dist/`
   - Configuration in `vercel.json` specifies build commands

2. **Backend Deployment** (If deploying to Vercel)
   - FastAPI app is wrapped in `api/index.py` as a serverless function
   - All `/api/*` routes are forwarded to this function
   - Python dependencies from `api/requirements.txt` are installed

3. **Routing**
   - `/` → Frontend static files
   - `/api/*` → Backend serverless function (or external URL)

## Prerequisites

1. **Vercel Account**
   - Sign up at [vercel.com](https://vercel.com/signup)
   - Connect your GitHub account

2. **Repository Access**
   - Ensure Vercel has access to your GitHub repository

---

## Deployment Option 1: Frontend-Only on Vercel (Recommended)

This is the recommended approach for the free tier, as the ML backend is too large for Vercel's serverless function limits.

### Step 1: Deploy Backend Elsewhere

Deploy the backend using one of these services:

**Option A: Railway** (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Note the deployment URL (e.g., https://your-app.railway.app)
```

**Option B: Render**
1. Go to [render.com](https://render.com)
2. Create new "Web Service"
3. Connect GitHub repository
4. Set:
   - Root Directory: `src/backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Option C: Docker Hosting** (Any provider)
```bash
# Build and push Docker image
cd src/backend
docker build -t your-registry/4dt907-backend .
docker push your-registry/4dt907-backend

# Deploy to your hosting provider
```

### Step 2: Deploy Frontend to Vercel

1. Import project to Vercel
2. Configure Project:
   - Framework: Other
   - Build Command: `cd src/frontend && npm install && npm run build`
   - Output Directory: `src/frontend/dist`
   
3. Set Environment Variable:
   ```
   BACKEND_URL=https://your-backend.railway.app
   ```

4. Deploy

This approach gives you:
- ✅ Free frontend hosting on Vercel
- ✅ No size limits on backend
- ✅ Better separation of concerns
- ✅ Independent scaling

---

## Deployment Option 2: Full Stack on Vercel Pro

For Vercel Pro users (250MB function limit).

### Method 1: Deploy via Vercel Dashboard (Recommended)

1. **Import Project**
   ```
   1. Go to https://vercel.com/dashboard
   2. Click "Add New" → "Project"
   3. Select your GitHub repository
   4. Click "Import"
   ```

2. **Configure Project**
   ```
   - Framework Preset: Other (or leave as detected)
   - Root Directory: ./
   - Build Command: (auto-detected from vercel.json)
   - Output Directory: src/frontend/dist
   ```

3. **Set Environment Variables**
   
   In Project Settings → Environment Variables, add:
   
   ```bash
   # Required for Backend
   BACKEND_PORT=8080
   
   # MLflow Configuration (if using MLflow)
   MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
   
   # Model URIs (configure based on your setup)
   MODEL_URI_PROD=your_production_model_uri
   MODEL_URI_DEV=your_development_model_uri
   MODEL_URI_BACKUP=your_backup_model_uri
   
   # Weakest Link Model URIs (if applicable)
   WEAKLINK_MODEL_URI_PROD=your_weaklink_prod_uri
   WEAKLINK_MODEL_URI_DEV=your_weaklink_dev_uri
   WEAKLINK_MODEL_URI_BACKUP=your_weaklink_backup_uri
   
   # CORS Configuration
   # Option 1: Allow specific pattern (recommended)
   ALLOWED_ORIGINS_PATTERN=^https://.*\.vercel\.app$
   
   # Option 2: Allow specific production domain
   PRODUCTION_URL=https://your-custom-domain.com
   ```

4. **Deploy**
   ```
   - Click "Deploy"
   - Wait for the build to complete
   - Your app will be available at https://your-project.vercel.app
   ```

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy from Project Root**
   ```bash
   cd /path/to/4dt907
   vercel
   ```

4. **Follow Prompts**
   - Set up and deploy: Y
   - Which scope: Select your account
   - Link to existing project: N (first time) or Y (subsequent deploys)
   - Project name: 4dt907
   - Directory: ./
   - Override settings: N

5. **Set Environment Variables**
   ```bash
   # Set each variable
   vercel env add BACKEND_PORT
   vercel env add MLFLOW_TRACKING_URI
   vercel env add MODEL_URI_PROD
   # ... etc
   ```

6. **Deploy to Production**
   ```bash
   vercel --prod
   ```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `BACKEND_PORT` | Port for backend (always 8080 on Vercel) | `8080` |

### Optional Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `https://mlflow.example.com` |
| `MODEL_URI_PROD` | Production model URI | `models:/MyModel@prod` |
| `MODEL_URI_DEV` | Development model URI | `models:/MyModel@dev` |
| `MODEL_URI_BACKUP` | Backup model URI | `models:/MyModel@backup` |
| `WEAKLINK_MODEL_URI_PROD` | Weakest link prod model | `models:/WeakLink@prod` |
| `WEAKLINK_MODEL_URI_DEV` | Weakest link dev model | `models:/WeakLink@dev` |
| `WEAKLINK_MODEL_URI_BACKUP` | Weakest link backup model | `models:/WeakLink@backup` |
| `ALLOWED_ORIGINS_PATTERN` | Regex for allowed CORS origins | `^https://.*\.vercel\.app$` |
| `PRODUCTION_URL` | Production domain URL | `https://app.example.com` |

## Testing Your Deployment

After deployment, test the following endpoints:

### 1. Frontend
```bash
curl https://your-project.vercel.app
```
Should return the frontend HTML.

### 2. Backend Health Check
```bash
curl https://your-project.vercel.app/api/health
```
Expected response:
```json
{"status": "ok"}
```

### 3. API Root
```bash
curl https://your-project.vercel.app/api/
```
Expected response:
```json
{
  "message": "Backend is running",
  "docs": "/docs",
  "health": "/health",
  ...
}
```

### 4. Prediction Endpoints
```bash
# Test prediction endpoint
curl -X POST https://your-project.vercel.app/api/v1/predict/latest \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, ...]}'
```

## Local Testing with Vercel Dev

To test the Vercel environment locally:

```bash
# Install Vercel CLI if not already installed
npm install -g vercel

# Run in development mode
vercel dev
```

This starts a local server that mimics the Vercel production environment:
- Frontend: http://localhost:3000
- Backend API: http://localhost:3000/api

## Troubleshooting

### Issue: Build Fails

**Problem**: Frontend build fails during deployment.

**Solution**:
1. Check build logs in Vercel dashboard
2. Ensure `src/frontend/package.json` has correct dependencies
3. Verify `vercel.json` buildCommand is correct
4. Test build locally: `cd src/frontend && npm run build`

### Issue: Backend API Returns 500 Error

**Problem**: API requests return internal server errors.

**Solution**:
1. Check Vercel function logs in dashboard
2. Verify environment variables are set correctly
3. Check that `api/requirements.txt` includes all dependencies
4. Verify `api/index.py` can import the FastAPI app

### Issue: CORS Errors

**Problem**: Frontend can't connect to backend due to CORS.

**Solution**:
1. Set `ALLOWED_ORIGINS_PATTERN` environment variable:
   ```
   ALLOWED_ORIGINS_PATTERN=^https://.*\.vercel\.app$
   ```
2. Or set specific `PRODUCTION_URL`:
   ```
   PRODUCTION_URL=https://your-project.vercel.app
   ```
3. Ensure backend CORS middleware is configured correctly

### Issue: Import Errors in Backend

**Problem**: Backend fails to import modules.

**Solution**:
1. Verify all Python dependencies are in `api/requirements.txt`
2. Check that `api/index.py` correctly adds `src/backend` to Python path
3. Ensure file structure matches expected paths

### Issue: Deployment Size Too Large

**Problem**: Vercel deployment fails with "Function size exceeds limit" error.

**Background**: Vercel serverless functions have a 50MB size limit (compressed). With ML dependencies (mlflow, numpy, pandas), the deployment can approach or exceed this limit.

**Solutions**:

1. **Use Vercel Pro Plan** (Recommended for ML applications)
   - Pro plan increases function size limit to 250MB
   - Suitable for ML models with heavy dependencies
   - [Upgrade here](https://vercel.com/pricing)

2. **Optimize Dependencies** (If staying on Free plan)
   - Remove unused dependencies from `api/requirements.txt`
   - Use lighter alternatives where possible
   - Consider pre-compiled models or model serving services

3. **Alternative Architecture** (For very large deployments)
   - Host backend separately (e.g., Railway, Render, or Docker)
   - Deploy only frontend to Vercel
   - Update `BACKEND_URL` environment variable to point to external backend
   - This maintains the same user experience with no frontend changes

**Current Deployment Size**:
- Core dependencies (FastAPI, uvicorn, python-dotenv): ~5-10MB
- ML dependencies (mlflow, numpy, pandas): ~40-60MB compressed
- Total: ~50-70MB (may exceed free tier limit)

**Recommendation**: For production ML applications, use Vercel Pro or host backend separately.

## Custom Domain

To use a custom domain:

1. Go to Project Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed by Vercel
4. Update `PRODUCTION_URL` environment variable
5. Update `ALLOWED_ORIGINS_PATTERN` or add domain to CORS settings

## Continuous Deployment

Vercel automatically deploys:
- **Production**: Pushes to `main` branch
- **Preview**: Pull requests and other branches

Configure in Project Settings → Git if needed.

## Monitoring

Monitor your deployment:
1. **Analytics**: Project → Analytics
2. **Logs**: Project → Logs (real-time function logs)
3. **Performance**: Project → Speed Insights

## Rollback

To rollback to a previous deployment:
1. Go to Project → Deployments
2. Find the working deployment
3. Click "..." → "Promote to Production"

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [FastAPI on Vercel](https://vercel.com/guides/deploying-fastapi-with-vercel)
- [Vite on Vercel](https://vercel.com/docs/frameworks/vite)

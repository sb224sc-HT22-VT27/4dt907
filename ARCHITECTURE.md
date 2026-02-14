# Project Architecture Overview

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Vercel Edge Network                       │
│                  (Global CDN + Routing)                      │
└────────────────────┬─────────────────┬──────────────────────┘
                     │                 │
        ┌────────────▼──────────┐     │
        │   Static Routes       │     │
        │   /, /assets/*        │     │
        └────────────┬──────────┘     │
                     │                 │
        ┌────────────▼──────────┐     │
        │   Frontend             │     │
        │   (React + Vite)       │     │
        │   src/frontend/dist/   │     │
        └────────────────────────┘     │
                                       │
                          ┌────────────▼──────────┐
                          │   API Routes          │
                          │   /api/*              │
                          └────────────┬──────────┘
                                       │
                          ┌────────────▼──────────┐
                          │   Backend             │
                          │   (FastAPI)           │
                          │   api/index.py        │
                          │   (Serverless)        │
                          └────────────┬──────────┘
                                       │
                          ┌────────────▼──────────┐
                          │   Business Logic      │
                          │   src/backend/app/    │
                          │   - API routes        │
                          │   - Services          │
                          │   - Models            │
                          └───────────────────────┘
```

## Local Development Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
│                  (Orchestration)                             │
└────────────────────┬─────────────────┬──────────────────────┘
                     │                 │
        ┌────────────▼──────────┐     │
        │   Frontend Container  │     │
        │   Port: 3030          │     │
        │   (Vite Dev Server)   │     │
        └────────────┬──────────┘     │
                     │                 │
                     │    ┌────────────▼──────────┐
                     │    │   Backend Container   │
                     └────►   Port: 8080          │
                          │   (FastAPI + Uvicorn) │
                          └────────────┬──────────┘
                                       │
                                       │
                          ┌────────────▼──────────┐
                          │   Shared Network      │
                          │   app-network         │
                          └───────────────────────┘
```

## File Structure Mapping

### Production (Vercel)
```
Request → Vercel
    ├── / → src/frontend/dist/index.html
    ├── /assets/* → src/frontend/dist/assets/*
    └── /api/* → api/index.py → src/backend/app/
```

### Development (Docker)
```
Request → Docker Compose
    ├── localhost:3030 → Frontend Container → src/frontend/
    └── localhost:8080 → Backend Container → src/backend/
```

## Environment-Based Configuration

### Vercel Environment
- **Build Time**: Runs in `src/frontend/` for frontend build
- **Runtime**: 
  - Static files served from CDN
  - API runs as serverless functions
- **CORS**: Uses `VERCEL_URL` and `PRODUCTION_URL`

### Docker Environment  
- **Build Time**: Docker builds containers from Dockerfiles
- **Runtime**: 
  - Frontend: Nginx serving Vite build
  - Backend: Uvicorn running FastAPI
- **CORS**: Uses `localhost:3030` and configured ports

## Request Flow Examples

### Frontend Request (Production)
```
User → https://app.vercel.app/
    → Vercel CDN
    → Static HTML from src/frontend/dist/
    → Browser renders React app
```

### API Request (Production)
```
User → https://app.vercel.app/api/health
    → Vercel Edge
    → api/index.py (Serverless Function)
    → src/backend/app/api/health.py
    → Response: {"status": "ok"}
```

### Full Stack Request (Production)
```
Browser → https://app.vercel.app/
    → Load React App (Static)
    → User clicks "Predict"
    → POST https://app.vercel.app/api/v1/predict/latest
    → api/index.py (Serverless)
    → src/backend/app/api/v1/endpoints/predict.py
    → Model Service → MLflow
    → Prediction Response
    → React displays result
```

## Development vs Production

| Aspect | Development (Docker) | Production (Vercel) |
|--------|---------------------|---------------------|
| **Frontend** | Vite dev server (HMR) | Static build (optimized) |
| **Backend** | Uvicorn (reload=True) | Serverless function |
| **Routing** | Vite proxy (/api) | Vercel rewrites |
| **CORS** | localhost:3030 | vercel.app domain |
| **Scaling** | Single instance | Auto-scaling |
| **Ports** | 3030, 8080 | N/A (HTTPS) |
| **SSL** | None (local) | Automatic |
| **Deploy** | `docker compose up` | `git push` |

## CI/CD Flow

```
Developer → git push
    ↓
GitHub
    ↓
┌───────────────────────────┐
│   GitHub Actions          │
│   - Backend tests         │
│   - Frontend lint         │
│   - Docker build test     │
└───────────┬───────────────┘
            ↓
    PR merged to main
            ↓
┌───────────┴───────────────┐
│   Vercel (Auto Deploy)    │
│   - Build frontend        │
│   - Deploy serverless API │
│   - Update production     │
└───────────────────────────┘
```

## File Organization

```
4dt907/
├── api/                     # Vercel serverless functions
│   ├── index.py            # Entry point (wraps FastAPI)
│   ├── requirements.txt    # Pinned dependencies
│   └── .python-version     # Python 3.9
│
├── src/
│   ├── backend/            # Backend source (Docker & Vercel)
│   │   ├── app/           # FastAPI application
│   │   │   ├── api/       # API endpoints
│   │   │   ├── models/    # Data models
│   │   │   ├── services/  # Business logic
│   │   │   └── main.py    # FastAPI app
│   │   ├── tests/         # Backend tests
│   │   ├── Dockerfile     # Docker build (dev)
│   │   └── requirements.txt
│   │
│   ├── frontend/           # Frontend source (Docker & Vercel)
│   │   ├── src/           # React components
│   │   ├── public/        # Static assets
│   │   ├── Dockerfile     # Docker build (dev)
│   │   ├── package.json   # Dependencies
│   │   └── vite.config.js # Vite config
│   │
│   └── docker-compose.yml # Local development
│
├── .github/workflows/      # CI/CD pipelines
├── vercel.json            # Vercel configuration
├── .vercelignore          # Vercel exclusions
├── README.md              # Main documentation
├── DEPLOYMENT.md          # Deployment guide
└── IMPLEMENTATION_SUMMARY.md  # This implementation
```

## Key Design Decisions

### 1. **Monorepo Structure**
- **Why**: Keeps all code in one repository
- **Benefit**: Easier to maintain, single source of truth
- **Trade-off**: Larger repo size

### 2. **Separate API Directory**
- **Why**: Vercel expects API functions in /api
- **Benefit**: Clean separation for deployment
- **Trade-off**: Small duplication (requirements.txt)

### 3. **Wrapper Pattern (api/index.py)**
- **Why**: Avoid restructuring existing code
- **Benefit**: 100% backward compatible
- **Trade-off**: Extra indirection layer

### 4. **Environment-Based CORS**
- **Why**: Support multiple deployment targets
- **Benefit**: Works in dev and prod
- **Trade-off**: Slightly more complex configuration

### 5. **Dual Documentation**
- **Why**: Support both Docker and Vercel users
- **Benefit**: Clear guidance for both approaches
- **Trade-off**: More documentation to maintain

## Migration Path

For teams currently using Docker:

1. ✅ **No immediate changes needed**
   - Continue using Docker for development
   - All existing workflows continue to work

2. ✅ **Optional Vercel adoption**
   - Deploy to Vercel when ready
   - No code changes required
   - Just configure environment variables

3. ✅ **Gradual transition**
   - Start with staging on Vercel
   - Keep production on Docker
   - Switch when confident

## Support and Resources

- **README.md**: Quick start and overview
- **DEPLOYMENT.md**: Detailed Vercel deployment guide
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details
- **This file**: Architecture and design overview

---

**Version**: 1.0
**Date**: February 14, 2026
**Status**: Production Ready ✅

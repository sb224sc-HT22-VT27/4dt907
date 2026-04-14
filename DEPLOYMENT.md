# Deployment

This document describes the deployment strategies for 4dt907.

## Overview

| Strategy | Frontend | Backend | When to use |
| -------- | -------- | ------- | ----------- |
| **Primary** | Vercel (static) | Render (web service) | Normal production |
| **Fallback** | Vercel (static) | Vercel (serverless) | If Render is unavailable |

---

## Primary Strategy: Vercel (frontend) + Render (backend)

The frontend is deployed as a static site on Vercel and the backend runs as a
long-lived web service on [Render](https://render.com).
This avoids serverless cold starts, execution time limits, and memory constraints
that can affect ML model serving.

### How it works

- The **frontend** is built from `src/frontend/` as a static site (`@vercel/static-build`).
- The **backend** runs the FastAPI app directly on Render via `uvicorn`.
- The frontend is built with `VITE_BACKEND_URL` pointing to the Render service so all
  `/api/*` calls go directly to Render (no server-side proxy needed on Vercel).

```Text
┌─────────────────────┐   VITE_BACKEND_URL   ┌─────────────────────┐
│       Vercel        │ ──────────────────►  │       Render        │
│  Frontend (static)  │                      │   Backend (FastAPI) │
└─────────────────────┘                      └─────────────────────┘
```

### Configuration

`vercel.json` at the repository root controls the Vercel build (frontend only):

```json
{
  "builds": [
    { "src": "src/frontend/package.json", "use": "@vercel/static-build",
      "config": { "distDir": "dist" } }
  ],
  "rewrites": [
    { "source": "/(.*)", "destination": "src/frontend/$1" }
  ]
}
```

`render.yaml` at the repository root is picked up by Render automatically when you
connect the repository:

```yaml
services:
  - type: web
    name: 4dt907-backend
    runtime: python
    rootDir: src/backend
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
```

### Step 1 – Deploy the backend on Render

1. Create a new **Web Service** on [render.com](https://render.com) and connect the
   repository. Render will detect `render.yaml` and pre-fill most settings.
2. Confirm the service settings:

   | Setting | Value |
   | ------- | ----- |
   | **Root directory** | `src/backend` |
   | **Runtime** | Python 3 |
   | **Build command** | `pip install -r requirements.txt` |
   | **Start command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
   | **Health check path** | `/health` |

3. Add the following environment variables in the Render dashboard:

   | Variable | Description |
   | -------- | ----------- |
   | `MLFLOW_TRACKING_URI` | MLflow / DagsHub tracking server URI |
   | `MODEL_URI_PROD` | Production model URI |
   | `MODEL_URI_DEV` | Development model URI |
   | `MODEL_URI_BACKUP` | Backup model URI |
   | `WEAKLINK_MODEL_URI_PROD` | Weakest-Link production model URI |
   | `WEAKLINK_MODEL_URI_DEV` | Weakest-Link development model URI |
   | `WEAKLINK_MODEL_URI_BACKUP` | Weakest-Link backup model URI |
   | `PRODUCTION_URL` | Your Vercel frontend URL (for CORS), e.g. `https://your-project.vercel.app` |

4. Deploy the service and note the external URL (e.g. `https://4dt907-backend.onrender.com`).

### Step 2 – Deploy the frontend on Vercel

1. Create (or update) a Vercel project connected to this repository.
2. In the Vercel project settings → **Environment Variables**, add:

   | Variable | Value |
   | -------- | ----- |
   | `VITE_BACKEND_URL` | `https://<your-service>.onrender.com` |
   | `VITE_SUPABASE_URL` | Your Supabase project URL (optional — enables keypoint storage) |
   | `VITE_SUPABASE_ANON_KEY` | Supabase anonymous key (optional — required when `VITE_SUPABASE_URL` is set) |

3. Trigger a new deployment (or let Vercel deploy automatically on push).
   The static build will bake `VITE_BACKEND_URL` into the bundle so every API call
   goes directly to Render.

---

## Fallback Strategy: Full Vercel Deployment

Use this strategy if Render is unavailable.
Both the frontend and a Python serverless function are deployed on Vercel.

> The serverless function may hit cold-start delays, execution time limits or
> memory constraints when loading large ML models.

### How it works

- The **frontend** is built as a static site.
- The **backend** is served as a serverless function via `api/index.py` (`@vercel/python`).
- Requests to `/api/*` are rewritten to the serverless function.

```Text
┌──────────────────────────────────────┐
│              Vercel                  │
│                                      │
│  ┌────────────┐   /api/*             │
│  │  Frontend  │ ──────────────────►  │
│  │  (static)  │          ┌────────┐  │
│  └────────────┘          │Backend │  │
│                          │(lambda)│  │
│         /*               └────────┘  │
│  ◄──────────────────────────────     │
└──────────────────────────────────────┘
```

### Restoring the full-Vercel `vercel.json`

Replace the contents of `vercel.json` with:

```json
{
  "version": 2,
  "builds": [
    { "src": "api/index.py", "use": "@vercel/python" },
    { "src": "src/frontend/package.json", "use": "@vercel/static-build",
      "config": { "distDir": "dist" } }
  ],
  "rewrites": [
    { "source": "/api/(.*)", "destination": "api/index.py" },
    { "source": "/(.*)",     "destination": "src/frontend/$1" }
  ]
}
```

### Environment variables (Vercel dashboard)

Set the following in the Vercel project settings under **Environment Variables**:

| Variable | Description |
| -------- | ----------- |
| `MLFLOW_TRACKING_URI` | MLflow / DagsHub tracking server URI |
| `MODEL_URI_PROD` | Production model URI |
| `MODEL_URI_DEV` | Development model URI |
| `MODEL_URI_BACKUP` | Backup model URI |
| `WEAKLINK_MODEL_URI_PROD` | Weakest-Link production model URI |
| `WEAKLINK_MODEL_URI_DEV` | Weakest-Link development model URI |
| `WEAKLINK_MODEL_URI_BACKUP` | Weakest-Link backup model URI |
| `VITE_SUPABASE_URL` | Supabase project URL (optional — enables keypoint storage) |
| `VITE_SUPABASE_ANON_KEY` | Supabase anonymous key (optional — required when URL is set) |

Leave `VITE_BACKEND_URL` empty – the frontend will use relative `/api/*` paths which
Vercel rewrites to the serverless function.

### Local preview (full-Vercel)

```bash
# Install Vercel CLI
npm i -g vercel

# From repository root
vercel dev
```

---

## Local Development (Docker)

For local full-stack development both services run via Docker Compose:

```bash
# From repository root – copy and fill in env vars first
cp .env.example src/.env

cd src
docker compose build
docker compose up -d
```

- Frontend: <http://localhost:3030>
- Backend API: <http://localhost:8080>
- API docs: <http://localhost:8080/docs>

See [src/backend/README.md](src/backend/README.md) and [src/frontend/README.md](src/frontend/README.md) for standalone development instructions.

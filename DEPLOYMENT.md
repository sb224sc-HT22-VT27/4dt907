# Deployment

This document describes how to run 4dt907 locally and how to deploy it to production.

## Overview

| Component | Service | Purpose |
| --------- | ------- | ------- |
| Frontend | Vercel | Static site hosting |
| Backend | Render | FastAPI web service |
| Database | Supabase | Squat keypoint storage |
| Model Registry | DagsHub | MLflow tracking & model registry |

---

## Local Development (Docker)

For local full-stack development both services run via Docker Compose.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.

### Setup

1. Copy the environment file and fill in the values:

   ```bash
   cp .env.example src/.env
   ```

2. Edit `src/.env` and set at minimum `MLFLOW_TRACKING_URI` and the `MODEL_URI_*`
   variables (see `.env.example` for descriptions of every variable).

3. Start all services:

   ```bash
   cd src
   docker compose up -d
   ```

| Service | URL |
| ------- | --- |
| Frontend | <http://localhost:3030> |
| Backend API | <http://localhost:8080> |
| API docs (Swagger) | <http://localhost:8080/docs> |

Hot-reload is enabled for both the backend (uvicorn `--reload`) and the frontend
(Vite dev server) when using `docker compose up` locally.

See [src/backend/README.md](src/backend/README.md) and
[src/frontend/README.md](src/frontend/README.md) for standalone (non-Docker)
development instructions.

---

## Split Hosting: Vercel + Render + Supabase + DagsHub

In production each concern is handled by a dedicated platform:

- **Vercel** hosts the static frontend.
- **Render** runs the FastAPI backend as a long-lived web service, avoiding serverless
  cold-start delays and memory constraints that affect ML model serving.
- **Supabase** stores squat keypoints submitted by users.
- **DagsHub** hosts the MLflow tracking server and model registry.

```Text
Browser
  │
  ├──► Vercel (static frontend)
  │         │  VITE_BACKEND_URL
  │         ▼
  │    Render (FastAPI backend)
  │         │  MLFLOW_TRACKING_URI
  │         ▼
  │    DagsHub (MLflow / model registry)
  │
  └──► Supabase (keypoint storage)
       VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY
```

### Step 1 – DagsHub (MLflow model registry)

DagsHub provides a hosted MLflow tracking server at no cost for public projects.

1. Create or log in to your [DagsHub](https://dagshub.com) account.
2. Create a repository connected to (or mirroring) this GitHub repo.
3. Navigate to **Settings → Integrations → MLflow** and copy the tracking URI:

   ```
   https://dagshub.com/<username>/<repo>.mlflow
   ```

4. Train your models and log them to the registry. Note the `runs:/<run-id>/model`
   or `models:/<name>@<alias>` URIs for each model variant you want to serve.

### Step 2 – Supabase (keypoint storage)

1. Create a free project on [supabase.com](https://supabase.com).
2. In the Supabase SQL editor, create the keypoints table and enable Row Level
   Security (RLS):

   ```sql
   CREATE TABLE squat_keypoints (
       id             SERIAL PRIMARY KEY,
       id_name        VARCHAR,
       raw_keypoints  JSONB NOT NULL,
       score          FLOAT8,
       classification VARCHAR,
       created_at     TIMESTAMPTZ DEFAULT now()
   );
   ```

3. Go to **Settings → API** and note:
   - **Project URL** → used as `VITE_SUPABASE_URL` in Vercel
   - **Anon (public) key** → used as `VITE_SUPABASE_ANON_KEY` in Vercel

### Step 3 – Deploy the backend on Render

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
   | `MLFLOW_TRACKING_URI` | DagsHub MLflow tracking URI (from Step 1) |
   | `MODEL_URI_PROD` | Production model URI |
   | `MODEL_URI_DEV` | Development model URI |
   | `MODEL_URI_BACKUP` | Backup model URI |
   | `WEAKLINK_MODEL_URI_PROD` | Weakest-Link production model URI |
   | `WEAKLINK_MODEL_URI_DEV` | Weakest-Link development model URI |
   | `WEAKLINK_MODEL_URI_BACKUP` | Weakest-Link backup model URI |
   | `Z_MODEL_URI_PROD` | Z-predictor production model URI |
   | `Z_MODEL_URI_DEV` | Z-predictor development model URI |
   | `Z_MODEL_URI_BACKUP` | Z-predictor backup model URI |
   | `PRODUCTION_URL` | Your Vercel frontend URL (for CORS), e.g. `https://your-project.vercel.app` |

4. Deploy the service and note the external URL (e.g. `https://4dt907-backend.onrender.com`).

### Step 4 – Deploy the frontend on Vercel

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

1. Create (or update) a Vercel project connected to this repository.
2. In the Vercel project settings → **Environment Variables**, add:

   | Variable | Value |
   | -------- | ----- |
   | `VITE_BACKEND_URL` | Render service URL from Step 3 (e.g. `https://4dt907-backend.onrender.com`) |
   | `VITE_SUPABASE_URL` | Supabase project URL from Step 2 |
   | `VITE_SUPABASE_ANON_KEY` | Supabase anon key from Step 2 |

3. Trigger a new deployment (or let Vercel deploy automatically on push).
   The static build will bake `VITE_BACKEND_URL` into the bundle so every API call
   goes directly to Render.

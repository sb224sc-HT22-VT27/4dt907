# Deployment

This document describes the deployment strategies for 4dt907.

## Overview

| Strategy | Frontend | Backend | When to use |
| -------- | -------- | ------- | ----------- |
| **Primary** | Vercel (static) | Vercel (serverless) | Normal production |
| **Backup** | Vercel (static) | Render (web service) | If serverless backend limitations are hit |

---

## Primary Strategy: Full Vercel Deployment

Both the frontend and backend are deployed together on Vercel using `vercel.json` at the repository root.

### How it works

- The **frontend** is built from `src/frontend/` as a static site (`@vercel/static-build`).
- The **backend** is served as a serverless function via `api/index.py` (`@vercel/python`), which wraps the FastAPI app.
- Requests to `/api/*` are rewritten to the serverless function; all other requests serve the frontend.

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

### Configuration

`vercel.json` at the repository root controls routing and build steps:

```json
{
  "builds": [
    { "src": "api/index.py",            "use": "@vercel/python" },
    { "src": "src/frontend/package.json","use": "@vercel/static-build",
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

Leave `BACKEND_URL` empty – the frontend will use the `/api` proxy automatically.

### Local preview

```bash
# Install Vercel CLI
npm i -g vercel

# From repository root
vercel dev
```

---

## Backup Strategy: Frontend on Vercel, Backend on Render

Use this strategy if serverless constraints (cold starts, execution time limits, memory) become a problem for the backend.

### How it works

- The **frontend** is still deployed on Vercel (static build only, no serverless function).
- The **backend** runs as a long-lived web service on [Render](https://render.com), directly serving the FastAPI app.
- The frontend is configured with `BACKEND_URL` pointing to the Render service.

```Text
┌─────────────────────┐      BACKEND_URL       ┌─────────────────────┐
│       Vercel        │ ─────────────────────► │       Render        │
│  Frontend (static)  │                        │   Backend (FastAPI) │
└─────────────────────┘                        └─────────────────────┘
```

### Setting up the backend on Render

1. Create a new **Web Service** on [render.com](https://render.com) and connect the repository.
2. Set the following in the Render service settings:

   | Setting | Value |
   | ------- | ----- |
   | **Root directory** | `src/backend` |
   | **Runtime** | Python 3 |
   | **Build command** | `pip install -r requirements.txt` |
   | **Start command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

3. Add the environment variables listed in the [Primary strategy](#environment-variables-vercel-dashboard) table to the Render service's environment settings.

### Configuring the frontend on Vercel

1. In the Vercel project settings, add (or update) the environment variable:

   | Variable | Value |
   | -------- | ----- |
   | `BACKEND_URL` | `https://<your-service>.onrender.com` |

2. Redeploy the frontend on Vercel for the change to take effect.

### Switching back to the primary strategy

1. Remove or clear `BACKEND_URL` in the Vercel environment variables.
2. Redeploy on Vercel.

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

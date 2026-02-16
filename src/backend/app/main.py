"""
FastAPI backend application for 4dt907 project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.api.v2.router import router as v2_router  # keep v2 scaffold

# Load env from:
# - current working dir .env (common when running from src/backend)
# - backend/.env

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
# - src/.env (common when running docker-compose from src/)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(title="4dt907 Backend API")

# Build allowed origins list
ALLOWED_ORIGINS = [
    f"http://localhost:{os.getenv('FRONTEND_PORT', '3030')}",
    "http://localhost:3000",
    "http://localhost:5173",
]

# Add Vercel deployment URLs if provided
vercel_url = os.getenv("VERCEL_URL")
if vercel_url:
    ALLOWED_ORIGINS.extend(
        [
            f"https://{vercel_url}",
            f"http://{vercel_url}",
        ]
    )

# Allow production domain if configured
production_url = os.getenv("PRODUCTION_URL")
if production_url:
    ALLOWED_ORIGINS.append(production_url)

HOST_PORT = int(os.getenv("BACKEND_PORT", "8080"))

# For Vercel deployment, we may need to allow all origins
# or use a wildcard pattern. In production, configure ALLOWED_ORIGINS_PATTERN
allow_origin_regex = os.getenv("ALLOWED_ORIGINS_PATTERN")

if allow_origin_regex:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.get("/status")
def root():
    return {
        "message": "Backend is running",
        "docs": "/docs",
        "health": "/health",
        "predict_champion": "/api/v1/predict/champion",
        "predict_latest": "/api/v1/predict/latest",
        "v2_status": "/api/v2/status",
    }


app.include_router(health_router)
app.include_router(v1_router, prefix="/api/v1")

# Keep v2 versioning available
app.include_router(v2_router, prefix="/api/v2")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=HOST_PORT, reload=True)

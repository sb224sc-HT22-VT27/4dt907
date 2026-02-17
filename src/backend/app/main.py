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
from app.api.v2.router import router as v2_router

root_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=root_path)

load_dotenv()

app = FastAPI(title="4dt907 Backend API")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3030",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
]

# Vercel automatically sets VERCEL_URL in production
vercel_url = os.getenv("VERCEL_URL")
if vercel_url:
    ALLOWED_ORIGINS.append(f"https://{vercel_url}")

# Ensure we include any custom production URL
if os.getenv("PRODUCTION_URL"):
    ALLOWED_ORIGINS.append(os.getenv("PRODUCTION_URL"))


HOST_PORT = int(os.getenv("BACKEND_PORT", "8080"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
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

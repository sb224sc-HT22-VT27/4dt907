"""
Vercel serverless function entry point for FastAPI backend.
This wraps the FastAPI app for Vercel deployment.
"""

import sys
from pathlib import Path

# Add the backend app directory to the Python path
backend_path = Path(__file__).resolve().parents[1] / "src" / "backend"
sys.path.insert(0, str(backend_path))

from app.main import app

# Vercel expects a variable named `app` or a function named `handler`
# FastAPI app can be used directly
handler = app

"""
Vercel serverless function entry point for FastAPI backend.
This wraps the FastAPI app for Vercel deployment.
"""

import sys
import os

# Get the backend directory (parent of api/)
api_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.dirname(api_dir)

# Add the backend directory to the path so we can import app.main
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from app.main import app  # noqa: F401
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise e

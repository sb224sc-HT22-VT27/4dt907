"""
Vercel serverless function entry point for FastAPI backend.
This wraps the FastAPI app for Vercel deployment.
"""

import sys
import os

# Add the backend directory to the path
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

parent_path = os.path.dirname(backend_path)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

try:
    from app.main import app  # noqa: F401
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise e

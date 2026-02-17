"""
Vercel serverless function entry point for FastAPI backend.
This wraps the FastAPI app for Vercel deployment.
"""

import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if root_path not in sys.path:
    sys.path.insert(0, root_path)

backend_path = os.path.join(root_path, "src", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from app.main import app
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise e
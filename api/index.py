"""
Vercel serverless function entry point for FastAPI backend.
This wraps the FastAPI app for Vercel deployment.
"""

import sys
import os

# Get the absolute path to the project root
# In Vercel, the current directory is usually the root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add project root and backend source to path
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Add the specific backend app directory
backend_path = os.path.join(root_path, "src", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Now import and export the app
# Vercel's Python runtime will automatically detect the 'app' variable
try:
    from app.main import app
except ImportError as e:
    # This helps debug if the path is still wrong
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise e

"""
FastAPI backend application for 4dt907 project.

This is a simple hello world API that serves as the foundation
for the ML-powered data-intensive system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

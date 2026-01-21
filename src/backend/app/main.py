"""
FastAPI backend application for 4dt907 project.

This is a simple hello world API that serves as the foundation
for the ML-powered data-intensive system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="4dt907 API",
    description="Backend API for ML-powered data-intensive system",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - Hello World."""
    return {"message": "Hello World from 4dt907 backend!"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "backend"}


@app.get("/api/v1/hello/{name}")
async def hello_name(name: str):
    """Personalized hello endpoint."""
    return {"message": f"Hello, {name}! Welcome to 4dt907."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

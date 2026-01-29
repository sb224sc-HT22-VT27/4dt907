"""
FastAPI backend application for 4dt907 project.

"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
#from app.api.v2.router import router as v2_router
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="4dt907 Backend API")

# For Assignment 1 simplicity (client-server demo): allow frontend calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Backend is running",
        "docs": "/docs",
        "hello_v1": "/api/v1/hello",
        "health": "/health",
    }

# Unversioned infra endpoint
app.include_router(health_router)

# Versioned API routers
app.include_router(v1_router, prefix="/api/v1")
#app.include_router(v2_router, prefix="/api/v2")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) #host=(HOST_PORT) OR BACKEND_PORT ??

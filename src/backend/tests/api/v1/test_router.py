from fastapi import FastAPI

from app.api.v1.router import router as v1_router


def test_v1_router_registers_expected_paths():
    app = FastAPI()
    app.include_router(v1_router, prefix="/api/v1")

    paths = {route.path for route in app.routes}

    assert "/api/v1/predict/champion" in paths
    assert "/api/v1/predict/latest" in paths
    assert "/api/v1/model-info/latest" in paths
    assert "/api/v1/model-info/champion" in paths

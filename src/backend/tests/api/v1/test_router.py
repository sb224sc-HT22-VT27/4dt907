from fastapi import FastAPI
from app.api.v1.router import router as v1_router

app = FastAPI()
app.include_router(v1_router, prefix="/api/v1")

paths = {route.path for route in app.routes}


def test_v1_router_registers_expected_paths_predict_champoin():
    assert "/api/v1/predict/champion" in paths


def test_v1_router_registers_expected_paths_predict_latest():
    assert "/api/v1/predict/latest" in paths


def test_v1_router_registers_expected_paths_model_info_champion():
    assert "/api/v1/model-info/champion" in paths


def test_v1_router_registers_expected_paths_model_info_latest():
    assert "/api/v1/model-info/latest" in paths


def test_v1_router_registers_expected_paths_model_info_weakest_link_champion():
    assert "/api/v1/model-info/weakest-link/champion" in paths


def test_v1_router_registers_expected_paths_model_info_weakest_link_latest():
    assert "/api/v1/model-info/weakest-link/latest" in paths


def test_v1_router_registers_expected_paths_weakest_link_champion():
    assert "/api/v1/weakest-link/champion" in paths


def test_v1_router_registers_expected_paths_weakest_link_latest():
    assert "/api/v1/weakest-link/latest" in paths

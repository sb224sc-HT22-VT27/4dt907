import pytest
from app.schemas.prediction import PredictRequest, PredictResponse

"""Test cases for prediction request and response schemas."""


def test_prediction_request():
    data = {
        "features": [0.5, 1.2, 3.4]
    }
    request = PredictRequest(**data)
    assert request.features == [0.5, 1.2, 3.4]


def test_predict_request_features_not_list():
    with pytest.raises(Exception):
        PredictRequest(features="not-a-list")


def test_predict_request_features_not_floats():
    with pytest.raises(Exception):
        PredictRequest(features=["a", "b", "c"])


def test_predict_response_valid():
    pass


def test_predict_response_missing_model_uri():
    with pytest.raises(Exception):
        PredictResponse(prediction=0.5)


def test_predict_response_invalid_types():
    with pytest.raises(Exception):
        PredictResponse(prediction="high", model_uri=123)

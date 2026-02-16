import pytest
from pydantic import ValidationError
from app.schemas.weakest_link import WeakestLinkResponse


class TestWeakestLinkResponse:
    """Test suite for WeakestLinkResponse schema"""

    def test_valid_model_creation_response_prediction(self):
        """Test creating a valid WeakestLinkResponse instance."""
        response = WeakestLinkResponse(
            prediction="test_prediction",
            model_uri="https://example.com/model"
        )
        assert response.prediction == "test_prediction"

    def test_valid_model_creation_response_model_uri(self):
        """Test creating a valid WeakestLinkResponse instance."""
        response = WeakestLinkResponse(
            prediction="test_prediction",
            model_uri="https://example.com/model"
        )
        assert response.model_uri == "https://example.com/model"

    def test_model_with_empty_strings_prediction(self):
        """Test that empty strings are accepted."""
        response = WeakestLinkResponse(
            prediction="",
            model_uri=""
        )
        assert response.prediction == ""

    def test_model_with_empty_strings_model_uri(self):
        """Test that empty strings are accepted."""
        response = WeakestLinkResponse(
            prediction="",
            model_uri=""
        )
        assert response.model_uri == ""

    def test_missing_prediction_field(self):
        """Test that missing prediction field raises ValidationError."""
        with pytest.raises(ValidationError):
            WeakestLinkResponse(model_uri="https://example.com/model")

    def test_missing_model_uri_field(self):
        """Test that missing model_uri field raises ValidationError."""
        with pytest.raises(ValidationError):
            WeakestLinkResponse(prediction="test")

    def test_both_fields_missing(self):
        """Test that missing both fields raises ValidationError."""
        with pytest.raises(ValidationError):
            WeakestLinkResponse()

    def test_extra_fields_ignored_prediction(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            extra_field="ignored"
        )
        assert response.prediction == "test"

    def test_extra_fields_ignored_model_uri(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            extra_field="ignored"
        )
        assert response.model_uri == "https://example.com/model"

    def test_extra_fields_ignored_extra_field(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            extra_field="ignored"
        )
        assert not hasattr(response, 'extra_field')

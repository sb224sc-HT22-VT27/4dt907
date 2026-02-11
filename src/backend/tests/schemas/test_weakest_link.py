import pytest
from pydantic import ValidationError
from app.schemas.weakest_link import WeakestLinkResponse


class TestWeakestLinkResponse:
    """Test suite for WeakestLinkResponse schema"""

    def test_valid_model_creation(self):
        """Test creating a valid WeakestLinkResponse instance."""
        response = WeakestLinkResponse(
            prediction="test_prediction",
            model_uri="https://example.com/model"
        )
        assert response.prediction == "test_prediction"
        assert response.model_uri == "https://example.com/model"

    def test_model_with_empty_strings(self):
        """Test that empty strings are accepted."""
        response = WeakestLinkResponse(
            prediction="",
            model_uri=""
        )
        assert response.prediction == ""
        assert response.model_uri == ""

    def test_missing_prediction_field(self):
        """Test that missing prediction field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WeakestLinkResponse(model_uri="https://example.com/model")

        assert "prediction" in str(exc_info.value)

    def test_missing_model_uri_field(self):
        """Test that missing model_uri field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WeakestLinkResponse(prediction="test")

        assert "model_uri" in str(exc_info.value)

    def test_both_fields_missing(self):
        """Test that missing both fields raises ValidationError."""
        with pytest.raises(ValidationError):
            WeakestLinkResponse()

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            extra_field="ignored"
        )
        # Extra field is ignored, not stored
        assert response.prediction == "test"
        assert response.model_uri == "https://example.com/model"
        assert not hasattr(response, 'extra_field')

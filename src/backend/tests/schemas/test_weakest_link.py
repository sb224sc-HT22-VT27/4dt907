import pytest
from pydantic import ValidationError
from app.schemas.weakest_link import WeakestLinkResponse


class TestWeakestLinkResponse:
    """Test suite for WeakestLinkResponse schema"""

    def test_valid_model_creation_response_prediction(self):
        """Test creating a valid WeakestLinkResponse instance."""
        response = WeakestLinkResponse(
            prediction="test_prediction",
            model_uri="https://example.com/model",
            run_id="run_12345",
        )
        assert response.prediction == "test_prediction"

    def test_valid_model_creation_response_model_uri(self):
        """Test creating a valid WeakestLinkResponse instance."""
        response = WeakestLinkResponse(
            prediction="test_prediction",
            model_uri="https://example.com/model",
            run_id="run_12345",
        )
        assert response.model_uri == "https://example.com/model"

    def test_valid_model_creation_response_run_id(self):
        """Test creating a valid WeakestLinkResponse instance."""
        response = WeakestLinkResponse(
            prediction="test_prediction",
            model_uri="https://example.com/model",
            run_id="run_12345",
        )
        assert response.run_id == "run_12345"

    def test_model_with_empty_strings_prediction(self):
        """Test that empty strings are accepted."""
        response = WeakestLinkResponse(
            prediction="", model_uri="model_uri", run_id="run_12345"
        )
        assert response.prediction == ""

    def test_model_with_empty_strings_model_uri(self):
        """Test that empty strings are accepted."""
        response = WeakestLinkResponse(
            prediction="0.1", model_uri="", run_id="run_12345"
        )
        assert response.model_uri == ""

    def test_model_with_empty_strings_run_id(self):
        """Test that empty strings are accepted."""
        response = WeakestLinkResponse(
            prediction="0.1", model_uri="model_uri", run_id=""
        )
        assert response.run_id == ""

    def test_missing_prediction_field(self):
        """Test that missing prediction field raises ValidationError."""
        with pytest.raises(ValidationError):
            WeakestLinkResponse(model_uri="https://example.com/model")

    def test_missing_model_uri_field(self):
        """Test that missing model_uri field raises ValidationError."""
        with pytest.raises(ValidationError):
            WeakestLinkResponse(prediction="test")

    def test_missing_run_id_field(self):
        """Test that missing run_id field does not raise ValidationError."""
        response = WeakestLinkResponse(
            prediction="test", model_uri="https://example.com/model"
        )
        assert response.run_id is None

    def test_both_fields_missing(self):
        """Test that missing both fields raises ValidationError."""
        with pytest.raises(ValidationError):
            WeakestLinkResponse()

    def test_extra_fields_ignored_prediction(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            run_id="run_12345",
            extra_field="ignored",
        )
        assert response.prediction == "test"

    def test_extra_fields_ignored_model_uri(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            run_id="run_12345",
            extra_field="ignored",
        )
        assert response.model_uri == "https://example.com/model"

    def test_extra_fields_ignored_run_id(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            run_id="run_12345",
            extra_field="ignored",
        )
        assert response.run_id == "run_12345"

    def test_extra_fields_ignored_extra_field(self):
        """Test that extra fields are ignored by default."""
        response = WeakestLinkResponse(
            prediction="test",
            model_uri="https://example.com/model",
            run_id="run_12345",
            extra_field="ignored",
        )
        assert not hasattr(response, "extra_field")

"""
Tests for FastAPI Endpoints

Tests API endpoints, rate limiting, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info."""
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data

    def test_liveness_probe(self, api_client):
        """Test liveness probe endpoint."""
        response = api_client.get("/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_probe(self, api_client):
        """Test readiness probe endpoint."""
        response = api_client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"


class TestPredictEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_single(self, api_client):
        """Test single prediction endpoint."""
        response = api_client.post(
            "/predict", json={"text": "This is a test input for prediction."}
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_text" in data
        assert "output" in data
        assert "model_version" in data
        assert "inference_time_ms" in data

    def test_predict_with_all_scores(self, api_client):
        """Test prediction with all scores."""
        response = api_client.post(
            "/predict",
            json={
                "text": "Test input",
                "return_all_scores": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    def test_predict_empty_text(self, api_client):
        """Test prediction with empty text fails validation."""
        response = api_client.post("/predict", json={"text": ""})

        assert response.status_code == 422  # Validation error

    def test_predict_text_too_long(self, api_client):
        """Test prediction with too long text fails validation."""
        response = api_client.post("/predict", json={"text": "x" * 10001})

        assert response.status_code == 422  # Validation error

    def test_predict_batch(self, api_client):
        """Test batch prediction endpoint."""
        response = api_client.post(
            "/predict/batch",
            json={
                "texts": [
                    "First test input",
                    "Second test input",
                    "Third test input",
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert "successful_count" in data
        assert "batch_time_ms" in data
        assert len(data["predictions"]) == 3

    def test_predict_batch_empty_list(self, api_client):
        """Test batch prediction with empty list fails."""
        response = api_client.post("/predict/batch", json={"texts": []})

        assert response.status_code == 422  # Validation error


class TestModelEndpoints:
    """Tests for model information endpoints."""

    def test_get_model_info(self, api_client):
        """Test model info endpoint."""
        response = api_client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "model_path" in data
        assert "task_type" in data
        assert "device" in data

    def test_list_model_versions(self, api_client):
        """Test model versions endpoint."""
        response = api_client.get("/model/versions")

        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert "latest_version" in data
        assert "metadata" in data


class TestMetricsEndpoints:
    """Tests for monitoring metrics endpoints."""

    def test_get_metrics(self, api_client):
        """Test metrics endpoint."""
        response = api_client.get("/metrics")

        assert response.status_code == 200

    def test_reset_metrics(self, api_client):
        """Test metrics reset endpoint."""
        response = api_client.post("/metrics/reset")

        assert response.status_code == 200


class TestErrorHandling:
    """Tests for API error handling."""

    def test_not_found_endpoint(self, api_client):
        """Test 404 for non-existent endpoint."""
        response = api_client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, api_client):
        """Test 405 for wrong HTTP method."""
        response = api_client.get("/predict")  # Should be POST

        assert response.status_code == 405

    def test_invalid_json(self, api_client):
        """Test error for invalid JSON."""
        response = api_client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestRequestValidation:
    """Tests for request validation."""

    def test_missing_required_field(self, api_client):
        """Test error when required field is missing."""
        response = api_client.post("/predict", json={})  # Missing 'text' field

        assert response.status_code == 422

    def test_invalid_field_type(self, api_client):
        """Test error for wrong field type."""
        response = api_client.post(
            "/predict",
            json={
                "text": "valid text",
                "return_all_scores": "not_a_boolean",
            },
        )

        assert response.status_code == 422

    def test_invalid_parameter_range(self, api_client):
        """Test error for out-of-range parameters."""
        response = api_client.post(
            "/predict",
            json={
                "text": "valid text",
                "temperature": 5.0,  # Max is 2.0
            },
        )

        assert response.status_code == 422

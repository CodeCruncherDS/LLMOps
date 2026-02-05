"""
Tests for Inference Module

Tests model inference, input validation, and error handling.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestLLMInference:
    """Tests for LLMInference class."""
    
    def test_validate_input_valid(self):
        """Test input validation with valid input."""
        from src.inference import InferenceInput
        
        input_data = InferenceInput(text="This is a valid test input.")
        
        assert input_data.text == "This is a valid test input."
        assert input_data.max_new_tokens == 256  # default
        assert input_data.temperature == 0.7  # default
    
    def test_validate_input_strip_whitespace(self):
        """Test that input text is stripped."""
        from src.inference import InferenceInput
        
        input_data = InferenceInput(text="  padded text  ")
        
        assert input_data.text == "padded text"
    
    def test_validate_input_empty(self):
        """Test validation fails for empty input."""
        from src.inference import InferenceInput
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            InferenceInput(text="")
    
    def test_validate_input_too_long(self):
        """Test validation fails for too long input."""
        from src.inference import InferenceInput
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            InferenceInput(text="x" * 10001)
    
    def test_validate_input_parameters(self):
        """Test parameter validation."""
        from src.inference import InferenceInput
        from pydantic import ValidationError
        
        # Valid parameters
        input_data = InferenceInput(
            text="test",
            max_new_tokens=100,
            temperature=1.0,
            top_p=0.95,
            top_k=100,
        )
        
        assert input_data.max_new_tokens == 100
        
        # Invalid temperature
        with pytest.raises(ValidationError):
            InferenceInput(text="test", temperature=3.0)
        
        # Invalid max_new_tokens
        with pytest.raises(ValidationError):
            InferenceInput(text="test", max_new_tokens=5000)
    
    def test_predict_classification(self, mock_inference_engine):
        """Test classification prediction."""
        result = mock_inference_engine.predict(
            text="This is a test input",
            return_all_scores=False,
        )
        
        assert result.input_text == "This is a test input"
        assert result.model_version == "test-v1.0.0"
        assert result.inference_time_ms > 0
        assert "predicted_class" in result.output
        assert "confidence" in result.output
    
    def test_predict_with_all_scores(self, mock_inference_engine):
        """Test prediction with all scores returned."""
        result = mock_inference_engine.predict(
            text="Test input",
            return_all_scores=True,
        )
        
        assert "all_scores" in result.output
    
    def test_get_model_info(self, mock_inference_engine):
        """Test getting model information."""
        info = mock_inference_engine.get_model_info()
        
        assert "model_version" in info
        assert "task_type" in info
        assert "device" in info
        assert info["model_version"] == "test-v1.0.0"
        assert info["task_type"] == "classification"
    
    def test_predict_batch(self, mock_inference_engine):
        """Test batch prediction."""
        texts = [
            "First test input",
            "Second test input",
            "Third test input",
        ]
        
        results = mock_inference_engine.predict_batch(texts, batch_size=2)
        
        assert len(results) == 3
        assert all(r.model_version == "test-v1.0.0" for r in results)


class TestInferenceOutput:
    """Tests for InferenceOutput model."""
    
    def test_inference_output_creation(self):
        """Test creating InferenceOutput."""
        from src.inference import InferenceOutput
        
        output = InferenceOutput(
            input_text="test input",
            output={"predicted_class": 1, "confidence": 0.95},
            model_version="v1.0.0",
            inference_time_ms=50.5,
            token_count=10,
        )
        
        assert output.input_text == "test input"
        assert output.output["predicted_class"] == 1
        assert output.model_version == "v1.0.0"
        assert output.inference_time_ms == 50.5


class TestInferenceError:
    """Tests for InferenceError exception."""
    
    def test_inference_error_creation(self):
        """Test creating InferenceError."""
        from src.inference import InferenceError
        
        original = ValueError("original error")
        error = InferenceError("Prediction failed", original_error=original)
        
        assert error.message == "Prediction failed"
        assert error.original_error is original
        assert str(error) == "Prediction failed"


class TestGetInferenceEngine:
    """Tests for cached inference engine factory."""
    
    def test_get_inference_engine_caching(self):
        """Test that inference engines are cached."""
        from src.inference import get_inference_engine
        
        # Clear the cache first
        get_inference_engine.cache_clear()
        
        with patch("src.inference.LLMInference") as MockEngine:
            mock_instance = MagicMock()
            MockEngine.return_value = mock_instance
            
            # First call creates new instance
            engine1 = get_inference_engine("classification")
            
            # Second call with same parameters returns cached
            engine2 = get_inference_engine("classification")
            
            assert engine1 is engine2
            assert MockEngine.call_count == 1
            
            # Different parameters creates new instance
            engine3 = get_inference_engine("generation")
            
            assert MockEngine.call_count == 2

"""
Test Configuration and Fixtures

Shared fixtures for pytest test suite.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_data(temp_dir: Path) -> dict:
    """Create sample test data."""
    data = {
        "texts": [
            "This is a positive review about a great product.",
            "I'm disappointed with the quality.",
            "The weather today is nice.",
            "This is a test sentence for classification.",
        ],
        "labels": [1, 0, 1, 1],  # 1=positive, 0=negative
    }
    
    # Save as CSV
    csv_path = temp_dir / "sample.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in zip(data["texts"], data["labels"]):
            writer.writerow([text, label])
    
    # Save as JSON
    json_path = temp_dir / "sample.json"
    json_data = [
        {"text": text, "label": label}
        for text, label in zip(data["texts"], data["labels"])
    ]
    with open(json_path, "w") as f:
        json.dump(json_data, f)
    
    data["csv_path"] = csv_path
    data["json_path"] = json_path
    
    return data


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token_id = 0
    
    # Mock __call__ method
    def tokenize(text, **kwargs):
        if isinstance(text, str):
            text = [text]
        return {
            "input_ids": [[1, 2, 3, 4, 5] for _ in text],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in text],
        }
    
    tokenizer.side_effect = tokenize
    tokenizer.__call__ = tokenize
    
    # Mock decode method
    tokenizer.decode.return_value = "Generated text output"
    
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""
    import torch
    
    model = MagicMock()
    model.config = MagicMock()
    model.config.vocab_size = 30522
    model.config.hidden_size = 768
    model.config.num_labels = 2
    model.config.pad_token_id = 0
    model.config.max_position_embeddings = 512
    
    # Mock forward pass for classification
    def forward(**kwargs):
        batch_size = 1
        if "input_ids" in kwargs:
            batch_size = len(kwargs["input_ids"])
        
        logits = torch.tensor([[0.1, 0.9]] * batch_size)
        return MagicMock(logits=logits)
    
    model.__call__ = forward
    model.side_effect = forward
    
    # Mock generate for text generation
    def generate(**kwargs):
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    model.generate = generate
    
    # Mock to() and eval()
    model.to.return_value = model
    model.eval.return_value = model
    
    return model


@pytest.fixture
def mock_inference_engine(mock_model, mock_tokenizer):
    """Create a mock inference engine."""
    from src.inference import LLMInference
    
    with patch.object(LLMInference, "_load_model"):
        engine = LLMInference(
            task_type="classification",
            enable_monitoring=False,
        )
        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        engine._model_version = "test-v1.0.0"
        
        yield engine


@pytest.fixture
def api_client(mock_inference_engine) -> Generator[TestClient, None, None]:
    """Create a test client for the API."""
    from src.api import app, lifespan
    import src.api as api_module
    
    # Mock the global inference engine
    original_engine = api_module.inference_engine
    api_module.inference_engine = mock_inference_engine
    
    with TestClient(app) as client:
        yield client
    
    # Restore original
    api_module.inference_engine = original_engine


@pytest.fixture
def mock_settings(temp_dir: Path):
    """Create mock settings for testing."""
    with patch("src.config.settings") as mock:
        mock.model.base_model_name = "distilbert-base-uncased"
        mock.model.fine_tuned_model_path = temp_dir / "models"
        mock.model.max_seq_length = 128
        mock.model.batch_size = 2
        mock.model.max_new_tokens = 50
        mock.model.temperature = 0.7
        mock.model.top_p = 0.9
        mock.model.top_k = 50
        
        mock.data.raw_data_path = temp_dir / "data" / "raw"
        mock.data.processed_data_path = temp_dir / "data" / "processed"
        mock.data.train_split = 0.8
        mock.data.validation_split = 0.1
        mock.data.test_split = 0.1
        mock.data.min_text_length = 5
        mock.data.max_text_length = 1000
        mock.data.random_seed = 42
        
        mock.logging.log_file = temp_dir / "logs" / "test.log"
        mock.logging.log_predictions = True
        mock.logging.log_errors = True
        
        mock.api.rate_limit_requests = 100
        mock.api.rate_limit_period = 60
        
        mock.environment = "development"
        mock.is_development = True
        mock.is_production = False
        mock.project_name = "Test LLM Pipeline"
        mock.project_version = "0.0.1-test"
        
        # Create directories
        (temp_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (temp_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (temp_dir / "models").mkdir(parents=True, exist_ok=True)
        (temp_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        yield mock


# Markers for different test categories
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")

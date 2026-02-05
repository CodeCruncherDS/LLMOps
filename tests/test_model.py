"""
Tests for Model Module

Tests model fine-tuning, versioning, and training utilities.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestModelVersionManager:
    """Tests for ModelVersionManager class."""

    def test_list_versions_empty(self, temp_dir):
        """Test listing versions when none exist."""
        from src.model import ModelVersionManager

        manager = ModelVersionManager(base_path=temp_dir / "models")

        versions = manager.list_versions()

        assert versions == []

    def test_list_versions_with_versions(self, temp_dir):
        """Test listing versions when they exist."""
        from src.model import ModelVersionManager

        models_dir = temp_dir / "models"
        (models_dir / "v1.0.0").mkdir(parents=True)
        (models_dir / "v1.0.1").mkdir()
        (models_dir / "v2.0.0").mkdir()

        manager = ModelVersionManager(base_path=models_dir)

        versions = manager.list_versions()

        assert len(versions) == 3
        assert versions[0] == "v2.0.0"  # Sorted descending

    def test_get_latest_version(self, temp_dir):
        """Test getting latest version."""
        from src.model import ModelVersionManager

        models_dir = temp_dir / "models"
        (models_dir / "v1.0.0").mkdir(parents=True)
        (models_dir / "v1.0.1").mkdir()

        manager = ModelVersionManager(base_path=models_dir)

        latest = manager.get_latest_version()

        assert latest == "v1.0.1"

    def test_get_latest_version_empty(self, temp_dir):
        """Test getting latest version when none exist."""
        from src.model import ModelVersionManager

        manager = ModelVersionManager(base_path=temp_dir / "models")

        latest = manager.get_latest_version()

        assert latest is None

    def test_create_new_version(self, temp_dir):
        """Test creating new version identifier."""
        from src.model import ModelVersionManager

        models_dir = temp_dir / "models"
        manager = ModelVersionManager(base_path=models_dir)

        # First version
        version = manager.create_new_version()
        assert version == "v1.0.0"

        # Create directory to simulate saved model
        (models_dir / "v1.0.0").mkdir(parents=True)

        # Second version
        version = manager.create_new_version()
        assert version == "v1.0.1"

    def test_save_and_get_metadata(self, temp_dir):
        """Test saving and retrieving metadata."""
        from src.model import ModelVersionManager

        manager = ModelVersionManager(base_path=temp_dir / "models")

        manager.save_metadata(
            version="v1.0.0",
            metrics={"accuracy": 0.95, "f1": 0.92},
            config={"learning_rate": 2e-5},
        )

        metadata = manager.get_metadata("v1.0.0")

        assert "created_at" in metadata
        assert metadata["metrics"]["accuracy"] == 0.95
        assert metadata["config"]["learning_rate"] == 2e-5


class TestLLMFineTuner:
    """Tests for LLMFineTuner class."""

    def test_init(self):
        """Test fine-tuner initialization."""
        from src.model import LLMFineTuner

        tuner = LLMFineTuner(
            model_name="distilbert-base-uncased",
            task_type="classification",
            num_labels=2,
        )

        assert tuner.model_name == "distilbert-base-uncased"
        assert tuner.task_type == "classification"
        assert tuner.num_labels == 2
        assert tuner.model is None  # Not loaded yet

    def test_get_device_cpu(self):
        """Test device detection falls back to CPU."""
        from src.model import LLMFineTuner

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                tuner = LLMFineTuner()
                device = tuner._get_device()

        assert device == "cpu"

    @pytest.mark.slow
    def test_load_model(self):
        """Test loading a model (slow test)."""
        from src.model import LLMFineTuner

        tuner = LLMFineTuner(
            model_name="distilbert-base-uncased",
            task_type="classification",
            num_labels=2,
            device="cpu",
        )

        model, tokenizer = tuner.load_model()

        assert model is not None
        assert tokenizer is not None

    def test_prepare_training_args(self, temp_dir):
        """Test preparing training arguments."""
        from src.model import LLMFineTuner

        tuner = LLMFineTuner()
        tuner.version_manager = MagicMock()
        tuner.version_manager.create_new_version.return_value = "v1.0.0"
        tuner.version_manager.get_version_path.return_value = temp_dir / "v1.0.0"

        args = tuner.prepare_training_args(
            num_epochs=3,
            batch_size=16,
            learning_rate=2e-5,
        )

        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 16
        assert args.learning_rate == 2e-5

    def test_compute_metrics_classification(self):
        """Test computing classification metrics."""
        import numpy as np

        from src.model import LLMFineTuner

        tuner = LLMFineTuner(task_type="classification")

        # Mock predictions: 3 samples, 2 classes
        predictions = np.array(
            [
                [0.1, 0.9],
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        )
        labels = np.array([1, 0, 1])

        eval_pred = MagicMock()
        eval_pred.__iter__ = lambda self: iter([predictions, labels])

        metrics = tuner.compute_metrics((predictions, labels))

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert metrics["accuracy"] == 1.0  # All correct


class TestMetricsCallback:
    """Tests for MetricsCallback."""

    def test_on_log(self):
        """Test logging training metrics."""
        from src.model import MetricsCallback

        callback = MetricsCallback()

        state = MagicMock()
        state.global_step = 100
        state.epoch = 1.5

        callback.on_log(None, state, None, logs={"loss": 0.5})

        assert len(callback.training_metrics) == 1
        assert callback.training_metrics[0]["loss"] == 0.5
        assert callback.training_metrics[0]["step"] == 100

    def test_on_evaluate(self):
        """Test logging evaluation metrics."""
        from src.model import MetricsCallback

        callback = MetricsCallback()

        state = MagicMock()
        state.global_step = 200
        state.epoch = 2.0

        callback.on_evaluate(None, state, None, metrics={"eval_loss": 0.3})

        assert len(callback.eval_metrics) == 1
        assert callback.eval_metrics[0]["eval_loss"] == 0.3

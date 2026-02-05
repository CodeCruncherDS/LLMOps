"""
Model Fine-tuning Module

Handles loading pre-trained models, fine-tuning pipelines, checkpointing,
evaluation metrics, and model versioning.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          PreTrainedModel, PreTrainedTokenizer, Trainer,
                          TrainingArguments)
from transformers.trainer_callback import TrainerCallback

from src.config import settings

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manages model versioning and metadata."""

    def __init__(self, base_path: Path = settings.model.fine_tuned_model_path):
        """
        Initialize version manager.

        Args:
            base_path: Base path for storing model versions.
        """
        self.base_path = Path(base_path)
        self.metadata_file = self.base_path / "version_metadata.json"

    def get_version_path(self, version: str) -> Path:
        """Get path for a specific version."""
        return self.base_path / version

    def list_versions(self) -> list[str]:
        """List all available model versions."""
        if not self.base_path.exists():
            return []

        versions = []
        for path in self.base_path.iterdir():
            if path.is_dir() and path.name.startswith("v"):
                versions.append(path.name)

        return sorted(versions, reverse=True)

    def get_latest_version(self) -> Optional[str]:
        """Get the latest model version."""
        versions = self.list_versions()
        return versions[0] if versions else None

    def create_new_version(self) -> str:
        """Create a new version identifier."""
        latest = self.get_latest_version()

        if latest is None:
            return "v1.0.0"

        # Parse version and increment patch number
        parts = latest[1:].split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        patch += 1

        return f"v{major}.{minor}.{patch}"

    def save_metadata(
        self,
        version: str,
        metrics: dict[str, float],
        config: dict[str, Any],
    ) -> None:
        """Save version metadata."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)

        metadata[version] = {
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "config": config,
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata for version {version}")

    def get_metadata(self, version: Optional[str] = None) -> dict[str, Any]:
        """Get metadata for a specific version or all versions."""
        if not self.metadata_file.exists():
            return {}

        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)

        if version:
            return metadata.get(version, {})
        return metadata


class MetricsCallback(TrainerCallback):
    """Custom callback for logging metrics during training."""

    def __init__(self):
        """Initialize metrics callback."""
        self.training_metrics: list[dict] = []
        self.eval_metrics: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics."""
        if logs:
            self.training_metrics.append(
                {"step": state.global_step, "epoch": state.epoch, **logs}
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics:
            self.eval_metrics.append(
                {"step": state.global_step, "epoch": state.epoch, **metrics}
            )


class LLMFineTuner:
    """
    Fine-tuning pipeline for Large Language Models.

    Supports sequence classification and causal language modeling tasks.
    """

    def __init__(
        self,
        model_name: str = settings.model.base_model_name,
        task_type: str = "classification",
        num_labels: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize the fine-tuner.

        Args:
            model_name: Hugging Face model identifier.
            task_type: Type of task ('classification' or 'generation').
            num_labels: Number of labels for classification.
            device: Device to use ('cuda', 'mps', or 'cpu').
        """
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.device = device or self._get_device()

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Trainer] = None
        self.version_manager = ModelVersionManager()

        logger.info(
            f"Initialized LLMFineTuner with model: {model_name}, device: {self.device}"
        )

    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load pre-trained model and tokenizer.

        Returns:
            Tuple of (model, tokenizer).
        """
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load appropriate model type
        if self.task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Update model config for padding token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.to(self.device)
        logger.info(f"Model loaded successfully on {self.device}")

        return self.model, self.tokenizer

    def prepare_training_args(
        self,
        output_dir: Optional[Path] = None,
        num_epochs: int = settings.model.num_epochs,
        batch_size: int = settings.model.batch_size,
        learning_rate: float = settings.model.learning_rate,
        warmup_steps: int = settings.model.warmup_steps,
        weight_decay: float = settings.model.weight_decay,
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        **kwargs: Any,
    ) -> TrainingArguments:
        """
        Prepare training arguments.

        Args:
            output_dir: Directory for saving checkpoints.
            num_epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            warmup_steps: Number of warmup steps.
            weight_decay: Weight decay for regularization.
            evaluation_strategy: When to evaluate ('epoch', 'steps').
            save_strategy: When to save checkpoints.
            load_best_model_at_end: Load best model after training.
            **kwargs: Additional training arguments.

        Returns:
            TrainingArguments instance.
        """
        version = self.version_manager.create_new_version()
        output_dir = output_dir or self.version_manager.get_version_path(version)

        # Determine device settings
        use_mps = self.device == "mps"
        use_cuda = self.device == "cuda"

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            save_total_limit=3,
            use_mps_device=use_mps,
            fp16=use_cuda,  # Enable fp16 for CUDA
            report_to=[],  # Disable external reporting
            **kwargs,
        )

        logger.info(f"Training arguments prepared for version: {version}")
        return args

    def compute_metrics(self, eval_pred) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Evaluation predictions from Trainer.

        Returns:
            Dictionary of metrics.
        """
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score)

        predictions, labels = eval_pred

        if self.task_type == "classification":
            preds = np.argmax(predictions, axis=1)

            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="weighted"),
                "precision": precision_score(labels, preds, average="weighted"),
                "recall": recall_score(labels, preds, average="weighted"),
            }

        # For generation tasks, compute perplexity
        loss = predictions.mean()
        perplexity = np.exp(loss) if loss < 100 else float("inf")
        return {"perplexity": perplexity}

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        early_stopping_patience: int = 3,
    ) -> dict[str, Any]:
        """
        Fine-tune the model.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            training_args: Training arguments.
            callbacks: Additional trainer callbacks.
            early_stopping_patience: Patience for early stopping.

        Returns:
            Training results including metrics.
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        training_args = training_args or self.prepare_training_args()

        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(self.tokenizer)

        # Setup callbacks
        metrics_callback = MetricsCallback()
        all_callbacks = [metrics_callback]

        if eval_dataset is not None:
            all_callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )

        if callbacks:
            all_callbacks.extend(callbacks)

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=all_callbacks,
        )

        logger.info("Starting training...")
        train_result = self.trainer.train()

        # Get final metrics
        metrics = train_result.metrics
        if eval_dataset:
            eval_metrics = self.trainer.evaluate()
            metrics.update(eval_metrics)

        # Save version metadata
        version = Path(training_args.output_dir).name
        self.version_manager.save_metadata(
            version=version,
            metrics=metrics,
            config={
                "model_name": self.model_name,
                "task_type": self.task_type,
                "num_labels": self.num_labels,
                "training_args": training_args.to_dict(),
            },
        )

        logger.info(f"Training completed. Metrics: {metrics}")

        return {
            "version": version,
            "metrics": metrics,
            "training_history": metrics_callback.training_metrics,
            "eval_history": metrics_callback.eval_metrics,
        }

    def save_model(
        self,
        output_path: Optional[Path] = None,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save the fine-tuned model.

        Args:
            output_path: Custom output path.
            version: Version identifier.

        Returns:
            Path where model was saved.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model to save. Train or load a model first.")

        version = version or self.version_manager.get_latest_version() or "v1.0.0"
        output_path = output_path or self.version_manager.get_version_path(version)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Model saved to {output_path}")
        return output_path

    def load_fine_tuned_model(
        self,
        version: Optional[str] = None,
        model_path: Optional[Path] = None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a fine-tuned model.

        Args:
            version: Version to load. If None, loads latest.
            model_path: Direct path to model. Overrides version.

        Returns:
            Tuple of (model, tokenizer).
        """
        if model_path is None:
            version = version or self.version_manager.get_latest_version()
            if version is None:
                raise ValueError("No fine-tuned model versions found")
            model_path = self.version_manager.get_version_path(version)

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading fine-tuned model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

        self.model.to(self.device)

        return self.model, self.tokenizer


def quick_train(
    train_data: Dataset,
    eval_data: Optional[Dataset] = None,
    model_name: str = settings.model.base_model_name,
    num_labels: int = 2,
    num_epochs: int = 3,
) -> dict[str, Any]:
    """
    Quick training function for simple use cases.

    Args:
        train_data: Training dataset (tokenized).
        eval_data: Evaluation dataset (tokenized).
        model_name: Model to fine-tune.
        num_labels: Number of labels.
        num_epochs: Number of epochs.

    Returns:
        Training results.
    """
    tuner = LLMFineTuner(
        model_name=model_name,
        task_type="classification",
        num_labels=num_labels,
    )

    tuner.load_model()

    training_args = tuner.prepare_training_args(num_epochs=num_epochs)

    results = tuner.train(
        train_dataset=train_data,
        eval_dataset=eval_data,
        training_args=training_args,
    )

    tuner.save_model(version=results["version"])

    return results

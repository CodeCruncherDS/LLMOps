"""
from __future__ import annotations

Model Inference Module

Handles model loading with caching, batch and single inference,
input validation, and error handling.
"""

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer, pipeline)

from src.config import settings
from src.monitor import PredictionMonitor

logger = logging.getLogger(__name__)


class InferenceInput(BaseModel):
    """Validated input for inference."""

    text: str = Field(..., min_length=1, max_length=10000)
    max_new_tokens: int = Field(default=settings.model.max_new_tokens, ge=1, le=4096)
    temperature: float = Field(default=settings.model.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default=settings.model.top_p, ge=0.0, le=1.0)
    top_k: int = Field(default=settings.model.top_k, ge=1, le=500)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Strip whitespace from text."""
        return v.strip()


class InferenceOutput(BaseModel):
    """Structured output from inference."""

    input_text: str
    output: Union[str, dict[str, Any]]
    model_version: str
    inference_time_ms: float
    token_count: Optional[int] = None


class InferenceError(Exception):
    """Custom exception for inference errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        """Initialize inference error."""
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class LLMInference:
    """
    Inference engine for Large Language Models.

    Supports both classification and generation tasks with
    caching, batch processing, and monitoring.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        task_type: str = "classification",
        device: Optional[str] = None,
        enable_monitoring: bool = True,
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the model. If None, uses config default.
            task_type: Type of task ('classification' or 'generation').
            device: Device to use ('cuda', 'mps', or 'cpu').
            enable_monitoring: Whether to enable prediction monitoring.
        """
        self.model_path = model_path or settings.model.fine_tuned_model_path
        self.task_type = task_type
        self.device = device or self._get_device()
        self.enable_monitoring = enable_monitoring

        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._pipeline = None
        self._model_version: str = "unknown"

        self.monitor = PredictionMonitor() if enable_monitoring else None

        logger.info(f"Initialized LLMInference: task={task_type}, device={self.device}")

    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> PreTrainedModel:
        """Lazy-load model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def model_version(self) -> str:
        """Get current model version."""
        return self._model_version

    def _load_model(self) -> None:
        """Load model and tokenizer from disk."""
        model_path = Path(self.model_path)
        model_source = None  # Will store the actual source to load from

        # Try to find the latest version if path is a directory with versions
        if model_path.exists() and model_path.is_dir():
            # Check if this is a version directory or contains versions
            version_dirs = [
                d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("v")
            ]
            if version_dirs:
                # Sort and get latest
                latest = sorted(version_dirs, reverse=True)[0]
                # Check if valid model exists in version dir
                if (latest / "config.json").exists():
                    model_source = latest
                    self._model_version = latest.name
            # Check if config.json exists directly in model_path
            elif (model_path / "config.json").exists():
                model_source = model_path
                self._model_version = settings.model.model_version

        # If no valid model found at path, use base model from HuggingFace
        if model_source is None:
            logger.info(
                f"No fine-tuned model found at {model_path}, "
                f"loading base model: {settings.model.base_model_name}"
            )
            model_source = settings.model.base_model_name
            self._model_version = "base"

        logger.info(f"Loading model from: {model_source}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_source)

            # Set padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            if self.task_type == "classification":
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_source
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(model_source)

            # Set pad token id in model config
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id

            self._model.to(self.device)
            self._model.eval()

            logger.info(f"Model loaded successfully. Version: {self._model_version}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise InferenceError(f"Failed to load model: {e}", original_error=e)

    def _create_pipeline(self):
        """Create HuggingFace pipeline for inference."""
        if self._pipeline is None:
            task = (
                "text-classification"
                if self.task_type == "classification"
                else "text-generation"
            )
            self._pipeline = pipeline(
                task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device in ["cuda", "mps"] else -1,
            )
        return self._pipeline

    def validate_input(self, text: str) -> InferenceInput:
        """
        Validate and normalize input text.

        Args:
            text: Input text to validate.

        Returns:
            Validated InferenceInput object.

        Raises:
            ValueError: If input is invalid.
        """
        try:
            return InferenceInput(text=text)
        except Exception as e:
            raise ValueError(f"Invalid input: {e}")

    def predict(
        self,
        text: str,
        return_all_scores: bool = False,
        **generation_kwargs: Any,
    ) -> InferenceOutput:
        """
        Make a single prediction.

        Args:
            text: Input text.
            return_all_scores: For classification, return all class scores.
            **generation_kwargs: Additional generation parameters.

        Returns:
            InferenceOutput with prediction results.
        """
        start_time = time.time()

        # Validate input
        validated_input = self.validate_input(text)

        try:
            result = self._predict_internal(
                validated_input,
                return_all_scores=return_all_scores,
                **generation_kwargs,
            )

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            output = InferenceOutput(
                input_text=validated_input.text,
                output=result,
                model_version=self._model_version,
                inference_time_ms=inference_time,
            )

            # Log prediction if monitoring enabled
            if self.monitor:
                self.monitor.log_prediction(
                    input_text=validated_input.text,
                    output=result,
                    model_version=self._model_version,
                    latency_ms=inference_time,
                )

            return output

        except Exception as e:
            inference_time = (time.time() - start_time) * 1000

            if self.monitor:
                self.monitor.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    input_text=text,
                )

            raise InferenceError(f"Prediction failed: {e}", original_error=e)

    def _predict_internal(
        self,
        validated_input: InferenceInput,
        return_all_scores: bool = False,
        **generation_kwargs: Any,
    ) -> Union[str, dict[str, Any]]:
        """Internal prediction logic."""
        if self.task_type == "classification":
            return self._predict_classification(
                validated_input.text,
                return_all_scores=return_all_scores,
            )
        else:
            return self._predict_generation(
                validated_input,
                **generation_kwargs,
            )

    def _predict_classification(
        self,
        text: str,
        return_all_scores: bool = False,
    ) -> dict[str, Any]:
        """Classification prediction."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.model.max_seq_length,
            padding=True,
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

        if return_all_scores:
            result["all_scores"] = {
                f"class_{i}": prob.item() for i, prob in enumerate(probabilities[0])
            }

        return result

    def _predict_generation(
        self,
        validated_input: InferenceInput,
        **kwargs: Any,
    ) -> str:
        """Text generation prediction."""
        # Tokenize
        inputs = self.tokenizer(
            validated_input.text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.model.max_seq_length,
        ).to(self.device)

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": validated_input.max_new_tokens,
            "temperature": validated_input.temperature,
            "top_p": validated_input.top_p,
            "top_k": validated_input.top_k,
            "do_sample": validated_input.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # Remove input text from output
        if generated_text.startswith(validated_input.text):
            generated_text = generated_text[len(validated_input.text) :].strip()

        return generated_text

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
        **kwargs: Any,
    ) -> list[InferenceOutput]:
        """
        Make batch predictions.

        Args:
            texts: List of input texts.
            batch_size: Batch size for processing.
            **kwargs: Additional prediction parameters.

        Returns:
            List of InferenceOutput objects.
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            for text in batch_texts:
                try:
                    result = self.predict(text, **kwargs)
                    results.append(result)
                except InferenceError as e:
                    logger.error(f"Batch prediction error for text: {e}")
                    # Create error result
                    results.append(
                        InferenceOutput(
                            input_text=text,
                            output={"error": str(e)},
                            model_version=self._model_version,
                            inference_time_ms=0,
                        )
                    )

        return results

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        # Ensure model is loaded
        _ = self.model

        return {
            "model_version": self._model_version,
            "model_path": str(self.model_path),
            "task_type": self.task_type,
            "device": self.device,
            "model_type": type(self._model).__name__,
            "vocab_size": self._model.config.vocab_size,
            "max_position_embeddings": getattr(
                self._model.config, "max_position_embeddings", None
            ),
            "hidden_size": getattr(self._model.config, "hidden_size", None),
            "num_labels": getattr(self._model.config, "num_labels", None),
        }

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model unloaded from memory")


@lru_cache(maxsize=4)
def get_inference_engine(
    task_type: str = "classification",
    model_path: Optional[str] = None,
) -> LLMInference:
    """
    Get a cached inference engine instance.

    Args:
        task_type: Type of task.
        model_path: Path to model.

    Returns:
        Cached LLMInference instance.
    """
    path = Path(model_path) if model_path else None
    return LLMInference(
        model_path=path,
        task_type=task_type,
    )

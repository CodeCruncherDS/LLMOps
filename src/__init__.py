"""
LLM Pipeline Project - Source Package

A production-ready MLOps workflow for Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "LLMOps Team"

from src.config import settings
from src.inference import LLMInference
from src.monitor import PredictionMonitor

__all__ = ["settings", "LLMInference", "PredictionMonitor", "__version__"]

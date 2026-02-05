"""
Tests for Data Loader Module

Tests data loading, preprocessing, and validation functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datasets import Dataset


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_csv(self, temp_dir, sample_data):
        """Test loading data from CSV file."""
        from src.data_loader import DataLoader

        loader = DataLoader()

        with patch.object(loader, "_tokenizer", MagicMock()):
            dataset = loader.load_csv(
                sample_data["csv_path"],
                text_column="text",
                label_column="label",
            )

        assert isinstance(dataset, Dataset)
        assert len(dataset) == len(sample_data["texts"])
        assert "text" in dataset.column_names

    def test_load_json(self, temp_dir, sample_data):
        """Test loading data from JSON file."""
        from src.data_loader import DataLoader

        loader = DataLoader()

        with patch.object(loader, "_tokenizer", MagicMock()):
            dataset = loader.load_json(
                sample_data["json_path"],
                text_column="text",
                label_column="label",
            )

        assert isinstance(dataset, Dataset)
        assert len(dataset) == len(sample_data["texts"])

    def test_load_csv_missing_column(self, temp_dir):
        """Test error when required column is missing."""
        from src.data_loader import DataLoader, DataValidationError

        csv_path = temp_dir / "missing_col.csv"
        pd.DataFrame({"wrong_column": ["text1", "text2"]}).to_csv(csv_path, index=False)

        loader = DataLoader()

        with pytest.raises(DataValidationError, match="Text column"):
            loader.load_csv(csv_path, text_column="text")

    def test_load_empty_csv(self, temp_dir):
        """Test error when CSV is empty."""
        from src.data_loader import DataLoader, DataValidationError

        csv_path = temp_dir / "empty.csv"
        pd.DataFrame(columns=["text", "label"]).to_csv(csv_path, index=False)

        loader = DataLoader()

        with pytest.raises(DataValidationError, match="empty"):
            loader.load_csv(csv_path)

    def test_preprocess_text(self):
        """Test text preprocessing."""
        from src.data_loader import DataLoader

        loader = DataLoader()

        # Test strip whitespace
        result = loader.preprocess_text("  hello world  ")
        assert result == "hello world"

        # Test lowercase
        result = loader.preprocess_text("Hello World", lowercase=True)
        assert result == "hello world"

        # Test min length filter
        result = loader.preprocess_text("hi", min_length=10)
        assert result is None

        # Test max length filter
        result = loader.preprocess_text("x" * 100, max_length=50)
        assert result is None

        # Test non-string input
        result = loader.preprocess_text(None)
        assert result is None

    def test_split_dataset(self, sample_data):
        """Test dataset splitting."""
        from src.data_loader import DataLoader

        loader = DataLoader()

        # Create a simple dataset
        dataset = Dataset.from_dict(
            {
                "text": sample_data["texts"] * 25,  # 100 samples
                "label": sample_data["labels"] * 25,
            }
        )

        splits = loader.split_dataset(
            dataset,
            train_size=0.8,
            validation_size=0.1,
            test_size=0.1,
        )

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

        total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total == 100

    def test_split_dataset_invalid_ratios(self, sample_data):
        """Test error when split ratios don't sum to 1."""
        from src.data_loader import DataLoader, DataValidationError

        loader = DataLoader()
        dataset = Dataset.from_dict(
            {
                "text": sample_data["texts"],
                "label": sample_data["labels"],
            }
        )

        with pytest.raises(DataValidationError, match="sum to 1.0"):
            loader.split_dataset(
                dataset,
                train_size=0.5,
                validation_size=0.3,
                test_size=0.3,  # Total = 1.1
            )


class TestCreateSampleDataset:
    """Tests for sample dataset creation."""

    def test_create_sample_dataset(self, temp_dir):
        """Test creating sample dataset."""
        from src.data_loader import create_sample_dataset

        output_path = temp_dir / "test_sample.json"

        with patch("src.data_loader.settings") as mock_settings:
            mock_settings.data.raw_data_path = temp_dir
            result_path = create_sample_dataset(
                num_samples=50,
                output_path=output_path,
            )

        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert len(data) == 50
        assert all("text" in item for item in data)
        assert all("label" in item for item in data)

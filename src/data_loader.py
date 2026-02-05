"""
Data Loading and Preprocessing Module

Handles data loading from various sources, preprocessing, tokenization,
and train/validation/test splitting with validation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config import settings

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataLoader:
    """
    Data loader for LLM training and inference.
    
    Supports loading from CSV, JSON, and Hugging Face datasets.
    Includes preprocessing, tokenization, and data splitting.
    """
    
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = settings.model.max_seq_length,
    ):
        """
        Initialize the data loader.
        
        Args:
            tokenizer: Pre-trained tokenizer. If None, loads from config.
            max_length: Maximum sequence length for tokenization.
        """
        self.max_length = max_length
        self._tokenizer = tokenizer
        
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model.base_model_name
            )
            # Set padding token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    def load_csv(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = "label",
        **kwargs: Any,
    ) -> Dataset:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file.
            text_column: Name of the text column.
            label_column: Name of the label column (optional).
            **kwargs: Additional arguments for pandas read_csv.
            
        Returns:
            Dataset: Hugging Face Dataset object.
            
        Raises:
            DataValidationError: If required columns are missing.
        """
        file_path = Path(file_path)
        logger.info(f"Loading CSV data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        df = pd.read_csv(file_path, **kwargs)
        return self._dataframe_to_dataset(df, text_column, label_column)
    
    def load_json(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = "label",
        **kwargs: Any,
    ) -> Dataset:
        """
        Load data from JSON/JSONL file.
        
        Args:
            file_path: Path to JSON file.
            text_column: Name of the text column.
            label_column: Name of the label column (optional).
            **kwargs: Additional arguments for pandas read_json.
            
        Returns:
            Dataset: Hugging Face Dataset object.
        """
        file_path = Path(file_path)
        logger.info(f"Loading JSON data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        # Determine if it's JSONL format
        if file_path.suffix == ".jsonl":
            df = pd.read_json(file_path, lines=True, **kwargs)
        else:
            df = pd.read_json(file_path, **kwargs)
            
        return self._dataframe_to_dataset(df, text_column, label_column)
    
    def load_huggingface_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        text_column: str = "text",
        label_column: Optional[str] = "label",
        **kwargs: Any,
    ) -> Union[Dataset, DatasetDict]:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.
            split: Specific split to load (train, validation, test).
            text_column: Name of the text column.
            label_column: Name of the label column (optional).
            **kwargs: Additional arguments for load_dataset.
            
        Returns:
            Dataset or DatasetDict: Loaded dataset.
        """
        logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        
        # Validate columns exist
        sample_dataset = dataset if isinstance(dataset, Dataset) else dataset["train"]
        self._validate_columns(sample_dataset, text_column, label_column)
        
        return dataset
    
    def _dataframe_to_dataset(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str],
    ) -> Dataset:
        """Convert pandas DataFrame to Hugging Face Dataset."""
        self._validate_dataframe(df, text_column, label_column)
        
        # Select relevant columns
        columns = [text_column]
        if label_column and label_column in df.columns:
            columns.append(label_column)
        
        df_subset = df[columns].copy()
        
        # Rename columns for consistency
        rename_map = {text_column: "text"}
        if label_column and label_column in df.columns:
            rename_map[label_column] = "label"
        df_subset = df_subset.rename(columns=rename_map)
        
        return Dataset.from_pandas(df_subset, preserve_index=False)
    
    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str],
    ) -> None:
        """Validate DataFrame has required columns and data."""
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        if text_column not in df.columns:
            raise DataValidationError(
                f"Text column '{text_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Check for null values in text column
        null_count = df[text_column].isnull().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in text column")
    
    def _validate_columns(
        self,
        dataset: Dataset,
        text_column: str,
        label_column: Optional[str],
    ) -> None:
        """Validate dataset has required columns."""
        if text_column not in dataset.column_names:
            raise DataValidationError(
                f"Text column '{text_column}' not found. "
                f"Available columns: {dataset.column_names}"
            )
    
    def preprocess_text(
        self,
        text: str,
        lowercase: bool = False,
        strip_whitespace: bool = True,
        min_length: int = settings.data.min_text_length,
        max_length: int = settings.data.max_text_length,
    ) -> Optional[str]:
        """
        Preprocess a single text string.
        
        Args:
            text: Input text.
            lowercase: Convert to lowercase.
            strip_whitespace: Strip leading/trailing whitespace.
            min_length: Minimum text length.
            max_length: Maximum text length.
            
        Returns:
            Preprocessed text or None if invalid.
        """
        if not isinstance(text, str):
            return None
        
        if strip_whitespace:
            text = text.strip()
        
        if lowercase:
            text = text.lower()
        
        # Length validation
        if len(text) < min_length or len(text) > max_length:
            return None
        
        return text
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        preprocessing_fn: Optional[Callable[[str], Optional[str]]] = None,
        num_proc: int = 4,
    ) -> Dataset:
        """
        Preprocess entire dataset.
        
        Args:
            dataset: Input dataset.
            text_column: Name of text column.
            preprocessing_fn: Custom preprocessing function.
            num_proc: Number of processes for parallel processing.
            
        Returns:
            Preprocessed dataset.
        """
        preprocess_fn = preprocessing_fn or self.preprocess_text
        
        def process_example(example: dict) -> dict:
            processed = preprocess_fn(example[text_column])
            example[text_column] = processed
            return example
        
        # Apply preprocessing
        dataset = dataset.map(process_example, num_proc=num_proc)
        
        # Filter out None values
        dataset = dataset.filter(
            lambda x: x[text_column] is not None,
            num_proc=num_proc
        )
        
        logger.info(f"Preprocessed dataset size: {len(dataset)}")
        return dataset
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        padding: str = "max_length",
        truncation: bool = True,
        num_proc: int = 4,
    ) -> Dataset:
        """
        Tokenize dataset for model training.
        
        Args:
            dataset: Input dataset.
            text_column: Name of text column.
            padding: Padding strategy.
            truncation: Whether to truncate sequences.
            num_proc: Number of processes for parallel processing.
            
        Returns:
            Tokenized dataset.
        """
        def tokenize_function(examples: dict) -> dict:
            return self.tokenizer(
                examples[text_column],
                padding=padding,
                truncation=truncation,
                max_length=self.max_length,
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )
        
        logger.info(f"Tokenized dataset with {len(tokenized)} examples")
        return tokenized
    
    def split_dataset(
        self,
        dataset: Dataset,
        train_size: float = settings.data.train_split,
        validation_size: float = settings.data.validation_split,
        test_size: float = settings.data.test_split,
        seed: int = settings.data.random_seed,
    ) -> DatasetDict:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Input dataset.
            train_size: Fraction for training.
            validation_size: Fraction for validation.
            test_size: Fraction for testing.
            seed: Random seed for reproducibility.
            
        Returns:
            DatasetDict with train, validation, and test splits.
        """
        # Validate splits sum to 1
        total = train_size + validation_size + test_size
        if abs(total - 1.0) > 0.001:
            raise DataValidationError(
                f"Split ratios must sum to 1.0, got {total}"
            )
        
        # First split: train vs (validation + test)
        train_test = dataset.train_test_split(
            train_size=train_size,
            seed=seed
        )
        
        # Second split: validation vs test
        val_test_ratio = test_size / (validation_size + test_size)
        val_test = train_test["test"].train_test_split(
            test_size=val_test_ratio,
            seed=seed
        )
        
        result = DatasetDict({
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        logger.info(
            f"Split dataset: train={len(result['train'])}, "
            f"validation={len(result['validation'])}, "
            f"test={len(result['test'])}"
        )
        
        return result
    
    def save_processed_data(
        self,
        dataset: Union[Dataset, DatasetDict],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Save processed dataset to disk.
        
        Args:
            dataset: Dataset to save.
            output_path: Output directory path.
            
        Returns:
            Path where dataset was saved.
        """
        output_path = output_path or settings.data.processed_data_path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(output_path))
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path
    
    def load_processed_data(
        self,
        input_path: Optional[Path] = None,
    ) -> Union[Dataset, DatasetDict]:
        """
        Load processed dataset from disk.
        
        Args:
            input_path: Input directory path.
            
        Returns:
            Loaded dataset.
        """
        from datasets import load_from_disk
        
        input_path = input_path or settings.data.processed_data_path
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found: {input_path}")
        
        dataset = load_from_disk(str(input_path))
        logger.info(f"Loaded processed data from {input_path}")
        
        return dataset


def create_sample_dataset(
    num_samples: int = 100,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Create a sample dataset for testing and demonstration.
    
    Args:
        num_samples: Number of samples to generate.
        output_path: Output file path.
        
    Returns:
        Path to created sample file.
    """
    import random
    
    output_path = output_path or settings.data.raw_data_path / "sample_data.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sample texts for demonstration
    sample_texts = [
        "This is a positive review about a great product.",
        "I'm disappointed with the quality of service.",
        "The weather today is absolutely beautiful!",
        "I need to complete my project by tomorrow.",
        "The new software update improved performance significantly.",
    ]
    
    labels = ["positive", "negative", "neutral"]
    
    data = []
    for i in range(num_samples):
        text = random.choice(sample_texts) + f" Sample {i + 1}."
        label = random.choice(labels)
        data.append({"text": text, "label": label})
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Created sample dataset with {num_samples} samples at {output_path}")
    return output_path

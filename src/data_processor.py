import pandas as pd
import os
import logging
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, tokenizer_name="facebook/mbart-large-50-many-to-many-mmt"):
        """
        Initialize data processor with tokenizer.
        
        Args:
            tokenizer_name (str): Pre-trained tokenizer name
        """
        logger.info(f"Initializing tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set default language codes
        self.hindi_code = "hi_IN"
        self.english_code = "en_XX"
    
    def load_data_from_csv(self, file_path, english_col="english", hindi_col="hindi"):
        """
        Load data from a CSV file containing English and Hindi text pairs.
        
        Args:
            file_path (str): Path to the CSV file
            english_col (str): Name of the column containing English text
            hindi_col (str): Name of the column containing Hindi text
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded data
        """
        logger.info(f"[testing] Loading data from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        if english_col not in df.columns or hindi_col not in df.columns:
            raise ValueError(f"Required columns {english_col} and/or {hindi_col} not found in the CSV file")
        
        logger.info(f"Loaded {len(df)} sentence pairs")
        return df
    
    def create_datasets(self, dataframe, english_col="english", hindi_col="hindi", 
                       val_split=0.1, test_split=0.1, seed=42):
        """
        Create train, validation, and test datasets from a DataFrame.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing English and Hindi text pairs
            english_col (str): Name of the column containing English text
            hindi_col (str): Name of the column containing Hindi text
            val_split (float): Proportion of data to use for validation
            test_split (float): Proportion of data to use for testing
            seed (int): Random seed for reproducibility
            
        Returns:
            DatasetDict: Dictionary containing train, validation, and test datasets
        """
        logger.info("[testing] Creating datasets from DataFrame")
        
        # Create a Hugging Face Dataset
        dataset = Dataset.from_pandas(dataframe)
        
        # Split the dataset
        splits = dataset.train_test_split(
            test_size=val_split + test_split,
            seed=seed
        )
        
        test_val_dataset = splits['test']
        train_dataset = splits['train']
        
        # Further split the test_val into test and validation
        test_val_splits = test_val_dataset.train_test_split(
            test_size=test_split / (val_split + test_split),
            seed=seed
        )
        
        # Create the final DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': test_val_splits['train'],
            'test': test_val_splits['test']
        })
        
        logger.info(f"Created datasets - Train: {len(dataset_dict['train'])}, "
                   f"Validation: {len(dataset_dict['validation'])}, "
                   f"Test: {len(dataset_dict['test'])} examples")
        
        return dataset_dict
    
    def preprocess_for_en_to_hi(self, examples, max_length=128):
        """
        Preprocess data for English to Hindi translation.
        
        Args:
            examples: Batch of examples
            max_length (int): Maximum sequence length
            
        Returns:
            dict: Preprocessed batch
        """
        # Set source language to English
        self.tokenizer.src_lang = self.english_code
        
        # Tokenize English inputs
        model_inputs = self.tokenizer(
            examples["english"], 
            max_length=max_length, 
            truncation=True, 
            padding="max_length"
        )
        
        # Set target language to Hindi
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["hindi"], 
                max_length=max_length, 
                truncation=True, 
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def preprocess_for_hi_to_en(self, examples, max_length=128):
        """
        Preprocess data for Hindi to English translation.
        
        Args:
            examples: Batch of examples
            max_length (int): Maximum sequence length
            
        Returns:
            dict: Preprocessed batch
        """
        # Set source language to Hindi
        self.tokenizer.src_lang = self.hindi_code
        
        # Tokenize Hindi inputs
        model_inputs = self.tokenizer(
            examples["hindi"], 
            max_length=max_length, 
            truncation=True, 
            padding="max_length"
        )
        
        # Set target language to English
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["english"], 
                max_length=max_length, 
                truncation=True, 
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_datasets(self, dataset_dict, direction="en_to_hi", max_length=128, batch_size=16):
        """
        Prepare datasets for model training.
        
        Args:
            dataset_dict (DatasetDict): Dictionary of datasets
            direction (str): Translation direction, either 'en_to_hi' or 'hi_to_en'
            max_length (int): Maximum sequence length
            batch_size (int): Batch size
            
        Returns:
            DatasetDict: Prepared datasets
        """
        logger.info(f"[testing] Preparing datasets for {direction} translation")
        
        # Select the appropriate preprocessing function
        if direction == "en_to_hi":
            preprocess_fn = self.preprocess_for_en_to_hi
        elif direction == "hi_to_en":
            preprocess_fn = self.preprocess_for_hi_to_en
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'en_to_hi' or 'hi_to_en'")
        
        # Apply preprocessing to all splits
        prepared_datasets = dataset_dict.map(
            preprocess_fn,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={"max_length": max_length}
        )
        
        # Set the format for PyTorch
        prepared_datasets = prepared_datasets.with_format("torch")
        
        logger.info("Datasets prepared successfully")
        return prepared_datasets 
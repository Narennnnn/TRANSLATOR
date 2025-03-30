#!/usr/bin/env python3
import argparse
import os
import logging
from src.data_processor import DataProcessor
from src.trainer import TranslationTrainer
from src.create_sample_dataset import create_sample_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Train the Hindi-English translator model."""
    parser = argparse.ArgumentParser(description="Train Hindi-English Translator Model")
    parser.add_argument("--data", "-d", help="Path to training data CSV")
    parser.add_argument("--output", "-o", default="./models", help="Output directory for trained model")
    parser.add_argument("--direction", choices=["en_to_hi", "hi_to_en"], default="en_to_hi",
                       help="Translation direction")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Create sample dataset if no data path is provided
    if args.data is None:
        logger.info("No data path provided, creating sample dataset")
        create_sample_dataset()
        args.data = "data/sample_dataset.csv"
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = data_processor.load_data_from_csv(args.data)
    
    # Create datasets
    dataset_dict = data_processor.create_datasets(df)
    
    # Prepare datasets for training
    prepared_datasets = data_processor.prepare_datasets(
        dataset_dict, 
        direction=args.direction,
        batch_size=args.batch_size
    )
    
    # Initialize trainer
    trainer = TranslationTrainer(output_dir=args.output)
    
    # Fine-tune the model
    logger.info(f"[testing] Fine-tuning model for {args.direction} translation")
    trainer.fine_tune(
        prepared_datasets["train"],
        prepared_datasets["validation"],
        direction=args.direction,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Evaluate on test set
    logger.info("[testing] Evaluating model on test set")
    results = trainer.evaluate_on_test(trainer, prepared_datasets["test"])
    
    # Print final path
    model_path = os.path.join(args.output, f"final-{args.direction}")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Test results: {results}")

if __name__ == "__main__":
    main() 
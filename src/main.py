import argparse
import logging
import pandas as pd
import os
from src.translator import HindiEnglishTranslator
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

def translate_text(text, direction="en_to_hi", model_path=None):
    """
    Translate a single text input.
    
    Args:
        text (str): Text to translate
        direction (str): Translation direction ('en_to_hi' or 'hi_to_en')
        model_path (str): Path to the fine-tuned model (if None, use the default model)
        
    Returns:
        str: Translated text
    """
    # Initialize the translator
    translator = HindiEnglishTranslator(model_name=model_path) if model_path else HindiEnglishTranslator()
    
    # Translate based on direction
    if direction == "en_to_hi":
        translation = translator.translate_en_to_hi(text)
    elif direction == "hi_to_en":
        translation = translator.translate_hi_to_en(text)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'en_to_hi' or 'hi_to_en'")
    
    return translation

def train_model(data_path=None, output_dir="./models", direction="en_to_hi", epochs=3):
    """
    Train a translation model.
    
    Args:
        data_path (str): Path to the training data CSV (if None, create a sample dataset)
        output_dir (str): Directory to save the fine-tuned model
        direction (str): Translation direction ('en_to_hi' or 'hi_to_en')
        epochs (int): Number of training epochs
        
    Returns:
        str: Path to the fine-tuned model
    """
    # Create sample dataset if no data path is provided
    if data_path is None:
        logger.info("No data path provided, creating sample dataset")
        create_sample_dataset()
        data_path = "data/sample_dataset.csv"
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = data_processor.load_data_from_csv(data_path)
    
    # Create datasets
    dataset_dict = data_processor.create_datasets(df)
    
    # Prepare datasets for training
    prepared_datasets = data_processor.prepare_datasets(dataset_dict, direction=direction)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = TranslationTrainer(output_dir=output_dir)
    
    # Fine-tune the model
    logger.info(f"[testing] Fine-tuning model for {direction} translation")
    trainer.fine_tune(
        prepared_datasets["train"],
        prepared_datasets["validation"],
        direction=direction,
        num_train_epochs=epochs
    )
    
    # Evaluate on test set
    logger.info("[testing] Evaluating model on test set")
    results = trainer.evaluate_on_test(trainer, prepared_datasets["test"])
    
    # Return the path to the fine-tuned model
    model_path = os.path.join(output_dir, f"final-{direction}")
    logger.info(f"Model saved to {model_path}")
    
    return model_path

def main():
    """Main function to run the translator."""
    parser = argparse.ArgumentParser(description="Hindi-English Translator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for translate command
    translate_parser = subparsers.add_parser("translate", help="Translate text")
    translate_parser.add_argument("text", help="Text to translate")
    translate_parser.add_argument("--direction", choices=["en_to_hi", "hi_to_en"], default="en_to_hi", 
                                 help="Translation direction")
    translate_parser.add_argument("--model", help="Path to the fine-tuned model")
    
    # Parser for train command
    train_parser = subparsers.add_parser("train", help="Train the translator model")
    train_parser.add_argument("--data", help="Path to the training data CSV")
    train_parser.add_argument("--output", default="./models", help="Directory to save the fine-tuned model")
    train_parser.add_argument("--direction", choices=["en_to_hi", "hi_to_en"], default="en_to_hi", 
                             help="Translation direction")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "translate":
        # Translate text
        translation = translate_text(args.text, args.direction, args.model)
        print(f"Original: {args.text}")
        print(f"Translation: {translation}")
    
    elif args.command == "train":
        # Train model
        model_path = train_model(args.data, args.output, args.direction, args.epochs)
        print(f"Model trained and saved to {model_path}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
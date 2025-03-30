#!/usr/bin/env python3
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_local():
    """Set up the Hindi-English Translator for local development."""
    logger.info("Setting up Hindi-English Translator for local development...")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Install dependencies
    logger.info("Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Install the package in development mode
    logger.info("Installing package in development mode...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'])
    
    # Download indic-nlp library
    if not os.path.exists('indic_nlp_library'):
        logger.info("Downloading indic-nlp library...")
        subprocess.run(['git', 'clone', 'https://github.com/anoopkunchukuttan/indic_nlp_library.git'])
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', 'indic_nlp_library'])
    
    # Create sample dataset
    if not os.path.exists('data/sample_dataset.csv'):
        logger.info("Creating sample dataset...")
        try:
            from src.create_sample_dataset import create_sample_dataset
            create_sample_dataset()
        except ImportError:
            logger.warning("Could not import create_sample_dataset. You can create it manually later.")
    
    logger.info("Setup completed!")
    logger.info("You can now use the translator locally.")
    logger.info("Example usage:")
    logger.info("from src.translator import HindiEnglishTranslator")
    logger.info("translator = HindiEnglishTranslator()")
    logger.info("hindi_text = translator.translate_en_to_hi('Hello, how are you?')")
    logger.info("print(hindi_text)")

def download_dataset(dataset_name):
    """
    Download a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download
    """
    logger.info(f"Downloading {dataset_name} dataset...")
    
    if dataset_name == 'iitb':
        # IIT Bombay English-Hindi Corpus
        data_dir = 'data/iitb'
        os.makedirs(data_dir, exist_ok=True)
        
        # Check if wget is available
        if os.name == 'nt':  # Windows
            logger.info("For Windows, please download the dataset manually from:")
            logger.info("http://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download/parallel.tgz")
            logger.info(f"Extract it to the {data_dir} directory")
        else:  # Unix-like
            subprocess.run(['wget', 'http://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download/parallel.tgz', '-P', data_dir])
            subprocess.run(['tar', '-xvzf', f'{data_dir}/parallel.tgz', '-C', data_dir])
        
        logger.info(f"IITB dataset downloaded to {data_dir}")
    
    elif dataset_name == 'wmt':
        # WMT Hindi-English dataset
        data_dir = 'data/wmt'
        os.makedirs(data_dir, exist_ok=True)
        
        # Download using Hugging Face datasets
        logger.info("Downloading WMT dataset using Hugging Face datasets...")
        subprocess.run([
            sys.executable, '-c', 
            'from datasets import load_dataset; dataset = load_dataset("wmt14", "hi-en"); dataset.save_to_disk("data/wmt")'
        ])
        
        logger.info(f"WMT dataset downloaded to {data_dir}")
    
    elif dataset_name == 'all':
        # Download all available datasets
        download_dataset('iitb')
        download_dataset('wmt')
    
    else:
        logger.error(f"Unknown dataset: {dataset_name}")

if __name__ == "__main__":
    setup_local()
    
    # If a dataset name is provided, download it
    if len(sys.argv) > 1:
        download_dataset(sys.argv[1]) 
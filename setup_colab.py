# Hindi-English Translator Setup for Google Colab
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

def setup_colab():
    """Setup the environment for Google Colab."""
    logger.info("Setting up Hindi-English Translator in Google Colab...")
    
    # Clone the repository if needed
    if not os.path.exists('Translator'):
        logger.info("Cloning repository...")
        subprocess.run(['git', 'clone', 'https://github.com/Narennnnn/TRANSLATOR.git', 'Translator'])
        os.chdir('Translator')
    elif os.path.exists('Translator'):
        os.chdir('Translator')
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # Install dependencies
    logger.info("Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Download indic-nlp library
    if not os.path.exists('indic_nlp_library'):
        logger.info("Downloading indic-nlp library...")
        subprocess.run(['git', 'clone', 'https://github.com/anoopkunchukuttan/indic_nlp_library.git'])
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', 'indic_nlp_library'])
    
    # Create sample dataset
    if not os.path.exists('data/sample_dataset.csv'):
        logger.info("Creating sample dataset...")
        from src.create_sample_dataset import create_sample_dataset
        create_sample_dataset()
    
    logger.info("Setup completed!")
    logger.info("You can now use the translator by importing the necessary modules.")
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
    setup_colab()
    
    # If a dataset name is provided, download it
    if len(sys.argv) > 1:
        download_dataset(sys.argv[1]) 
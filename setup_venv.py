#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
import platform
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

VENV_DIR = "venv"

def create_venv():
    """Create a virtual environment."""
    logger.info(f"Creating virtual environment in './{VENV_DIR}'...")
    
    # Check if venv already exists
    if os.path.exists(VENV_DIR):
        logger.info(f"Virtual environment already exists in './{VENV_DIR}'")
        return True
    
    try:
        # Create the virtual environment
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        logger.info(f"Virtual environment created successfully in './{VENV_DIR}'")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def get_venv_pip():
    """Get the path to the pip executable in the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "pip")

def setup_venv():
    """Set up the Hindi-English Translator with a virtual environment."""
    logger.info("Setting up Hindi-English Translator with virtual environment...")
    
    # Create virtual environment
    if not create_venv():
        logger.error("Failed to set up virtual environment. Exiting.")
        return False
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    venv_python = get_venv_python()
    venv_pip = get_venv_pip()
    
    # Check if virtual environment executables exist
    if not (os.path.exists(venv_python) and os.path.exists(venv_pip)):
        logger.error(f"Virtual environment executables not found. Check if '{VENV_DIR}' is properly created.")
        return False
    
    # Install dependencies
    logger.info("Installing dependencies in virtual environment...")
    subprocess.run([venv_pip, "install", "-U", "pip"], check=True)
    subprocess.run([venv_pip, "install", "-r", "requirements.txt"], check=True)
    
    # Install the package in development mode
    logger.info("Installing package in development mode...")
    subprocess.run([venv_pip, "install", "-e", "."], check=True)
    
    # Download indic-nlp library
    if not os.path.exists('indic_nlp_library'):
        logger.info("Downloading indic-nlp library...")
        subprocess.run(['git', 'clone', 'https://github.com/anoopkunchukuttan/indic_nlp_library.git'], check=True)
        subprocess.run([venv_pip, "install", "-e", "indic_nlp_library"], check=True)
    
    # Create sample dataset
    if not os.path.exists('data/sample_dataset.csv'):
        logger.info("Creating sample dataset...")
        subprocess.run([venv_python, "-c", "from src.create_sample_dataset import create_sample_dataset; create_sample_dataset()"], check=True)
    
    # Create activation scripts
    create_activation_scripts()
    
    logger.info("Setup completed successfully!")
    logger.info("\nTo use the translator:")
    if platform.system() == "Windows":
        logger.info("1. Run 'activate.bat' to activate the virtual environment")
    else:
        logger.info("1. Run 'source activate.sh' to activate the virtual environment")
    logger.info("2. Then you can use any of the provided scripts:")
    logger.info("   - python quick_test.py")
    logger.info("   - python run_translator.py --text \"Hello, how are you?\"")
    logger.info("   - python run_web_interface.py")
    
    return True

def create_activation_scripts():
    """Create activation scripts for different platforms."""
    # Create shell script for Unix-like systems
    if platform.system() != "Windows":
        with open("activate.sh", "w") as f:
            f.write(f"""#!/bin/bash
# Activate virtual environment for Hindi-English Translator
source {VENV_DIR}/bin/activate
echo "Virtual environment activated. You can now run the translator scripts."
echo "Use 'deactivate' when you're done to exit the virtual environment."
""")
        os.chmod("activate.sh", 0o755)
        logger.info("Created 'activate.sh' script")
    
    # Create batch script for Windows
    if platform.system() == "Windows":
        with open("activate.bat", "w") as f:
            f.write(f"""@echo off
:: Activate virtual environment for Hindi-English Translator
call {VENV_DIR}\\Scripts\\activate.bat
echo Virtual environment activated. You can now run the translator scripts.
echo Use 'deactivate' when you're done to exit the virtual environment.
""")
        logger.info("Created 'activate.bat' script")

def download_dataset(dataset_name):
    """
    Download a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset to download
    """
    logger.info(f"Downloading {dataset_name} dataset...")
    
    venv_python = get_venv_python()
    
    if dataset_name == 'iitb':
        # IIT Bombay English-Hindi Corpus
        data_dir = 'data/iitb'
        os.makedirs(data_dir, exist_ok=True)
        
        # Check if wget is available
        if platform.system() == "Windows":
            logger.info("For Windows, please download the dataset manually from:")
            logger.info("http://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download/parallel.tgz")
            logger.info(f"Extract it to the {data_dir} directory")
        else:
            # Check if wget is installed
            if shutil.which("wget"):
                subprocess.run(['wget', 'http://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download/parallel.tgz', '-P', data_dir], check=True)
                subprocess.run(['tar', '-xvzf', f'{data_dir}/parallel.tgz', '-C', data_dir], check=True)
            else:
                logger.error("wget is not installed. Please install wget or download the dataset manually.")
        
        logger.info(f"IITB dataset downloaded to {data_dir}")
    
    elif dataset_name == 'wmt':
        # WMT Hindi-English dataset
        data_dir = 'data/wmt'
        os.makedirs(data_dir, exist_ok=True)
        
        # Download using Hugging Face datasets
        logger.info("Downloading WMT dataset using Hugging Face datasets...")
        subprocess.run([
            venv_python, '-c', 
            'from datasets import load_dataset; dataset = load_dataset("wmt14", "hi-en"); dataset.save_to_disk("data/wmt")'
        ], check=True)
        
        logger.info(f"WMT dataset downloaded to {data_dir}")
    
    elif dataset_name == 'all':
        # Download all available datasets
        download_dataset('iitb')
        download_dataset('wmt')
    
    else:
        logger.error(f"Unknown dataset: {dataset_name}")

if __name__ == "__main__":
    try:
        setup_venv()
        
        # If a dataset name is provided, download it
        if len(sys.argv) > 1:
            download_dataset(sys.argv[1])
    except Exception as e:
        logger.error(f"An error occurred during setup: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
Script to evaluate the fine-tuned model's performance on idioms.
This script compares the base model's translations with the fine-tuned model.
"""

import argparse
import csv
import os
import sys
import logging
from tqdm import tqdm
import pandas as pd

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.translator import HindiEnglishTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_idioms(csv_path):
    """Load idioms from a CSV file."""
    idioms = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            idioms.append({
                "english": row['english'],
                "hindi": row['hindi']
            })
    return idioms

def evaluate_translations(idioms, base_model, fine_tuned_model=None, direction="en_to_hi"):
    """Evaluate and compare translations from base and fine-tuned models."""
    results = []
    
    # Initialize translators
    base_translator = HindiEnglishTranslator(model_name=base_model)
    fine_tuned_translator = HindiEnglishTranslator(model_name=fine_tuned_model) if fine_tuned_model else None
    
    logger.info(f"[testing] Evaluating translations for {len(idioms)} idioms ({direction})")
    
    for idiom in tqdm(idioms, desc="Translating"):
        if direction == "en_to_hi":
            source_text = idiom["english"]
            expected = idiom["hindi"]
            
            # Translate with base model
            base_translation = base_translator.translate_en_to_hi(source_text)
            
            # Translate with fine-tuned model (if available)
            fine_tuned_translation = fine_tuned_translator.translate_en_to_hi(source_text) if fine_tuned_translator else None
            
        else:  # hi_to_en
            source_text = idiom["hindi"]
            expected = idiom["english"]
            
            # Translate with base model
            base_translation = base_translator.translate_hi_to_en(source_text)
            
            # Translate with fine-tuned model (if available)
            fine_tuned_translation = fine_tuned_translator.translate_hi_to_en(source_text) if fine_tuned_translator else None
        
        # Add to results
        result = {
            "source": source_text,
            "expected": expected,
            "base_translation": base_translation,
            "fine_tuned_translation": fine_tuned_translation
        }
        results.append(result)
    
    return results

def save_results(results, output_path):
    """Save evaluation results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"[testing] Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance on idioms")
    parser.add_argument("--csv", default="data/custom_idioms/idioms.csv", help="Path to CSV file with idioms")
    parser.add_argument("--base-model", default="facebook/mbart-large-50-many-to-many-mmt", help="Base model for comparison")
    parser.add_argument("--fine-tuned-model", help="Fine-tuned model for evaluation")
    parser.add_argument("--direction", choices=["en_to_hi", "hi_to_en"], default="en_to_hi", help="Translation direction")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output file for evaluation results")
    
    args = parser.parse_args()
    
    # Load idioms
    idioms = load_idioms(args.csv)
    logger.info(f"Loaded {len(idioms)} idioms from {args.csv}")
    
    # Evaluate translations
    results = evaluate_translations(
        idioms,
        args.base_model,
        args.fine_tuned_model,
        args.direction
    )
    
    # Save results
    save_results(results, args.output)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main() 
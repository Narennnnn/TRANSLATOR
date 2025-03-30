#!/usr/bin/env python3
"""
Script to prepare a dataset of idioms for fine-tuning the translation model.
This script takes a CSV file of idioms and creates training/validation datasets.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

def read_idioms_csv(csv_path):
    """Read idioms from a CSV file."""
    idioms_en_to_hi = []
    idioms_hi_to_en = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Add English to Hindi pair
            idioms_en_to_hi.append({
                "input": row['english'],
                "target": row['hindi']
            })
            
            # Add Hindi to English pair
            idioms_hi_to_en.append({
                "input": row['hindi'],
                "target": row['english']
            })
    
    return idioms_en_to_hi, idioms_hi_to_en

def prepare_dataset(idioms, direction, output_dir, train_ratio=0.8):
    """Prepare training and validation datasets."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle the idioms
    random.shuffle(idioms)
    
    # Split into training and validation sets
    split_idx = int(len(idioms) * train_ratio)
    train_data = idioms[:split_idx]
    val_data = idioms[split_idx:]
    
    # Add language tags for mBART
    if direction == "en_to_hi":
        for item in train_data + val_data:
            item["source_lang"] = "en_XX"
            item["target_lang"] = "hi_IN"
    else:  # hi_to_en
        for item in train_data + val_data:
            item["source_lang"] = "hi_IN"
            item["target_lang"] = "en_XX"
    
    # Write train and validation data
    train_path = os.path.join(output_dir, f"train-{direction}.json")
    val_path = os.path.join(output_dir, f"val-{direction}.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created {direction} datasets:")
    print(f"  - Training: {len(train_data)} samples -> {train_path}")
    print(f"  - Validation: {len(val_data)} samples -> {val_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare idiom dataset for fine-tuning")
    parser.add_argument("--csv", default="data/custom_idioms/idioms.csv", help="Path to CSV file with idioms")
    parser.add_argument("--output-dir", default="data/fine_tuning", help="Output directory for prepared datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training to validation data")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Read idioms from CSV
    print(f"Reading idioms from {args.csv}")
    idioms_en_to_hi, idioms_hi_to_en = read_idioms_csv(args.csv)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare datasets for both directions
    prepare_dataset(idioms_en_to_hi, "en_to_hi", args.output_dir, args.train_ratio)
    prepare_dataset(idioms_hi_to_en, "hi_to_en", args.output_dir, args.train_ratio)
    
    print("\nDataset preparation complete!")
    print(f"Total idioms: {len(idioms_en_to_hi)}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script to fix language tags in the fine-tuning datasets.
This script ensures that the source_lang and target_lang match the actual content.
"""

import json
import os
import re

def is_hindi(text):
    """Check if the text contains Hindi characters."""
    # Hindi Unicode range (approximate)
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(hindi_pattern.search(text))

def fix_dataset(input_file, output_file, direction):
    """Fix language tags in the dataset."""
    fixed_data = []
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # Skip placeholder entries
            if item["input"] == "english" or len(item["input"].strip()) == 0 or len(item["target"].strip()) == 0:
                skipped += 1
                continue
            
            # Fix direction
            if direction == "en_to_hi":
                # Check if input is actually Hindi
                if is_hindi(item["input"]):
                    # Swap input and target
                    item["input"], item["target"] = item["target"], item["input"]
                # Ensure correct language tags
                item["source_lang"] = "en_XX"
                item["target_lang"] = "hi_IN"
            else:  # hi_to_en
                # Check if input is actually English
                if not is_hindi(item["input"]):
                    # Swap input and target
                    item["input"], item["target"] = item["target"], item["input"]
                # Ensure correct language tags
                item["source_lang"] = "hi_IN"
                item["target_lang"] = "en_XX"
            
            fixed_data.append(item)
    
    # Write fixed data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in fixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(fixed_data), skipped

def main():
    data_dir = "data/fine_tuning"
    output_dir = "data/fine_tuning_fixed"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix all datasets
    datasets = [
        ("train-en_to_hi.json", "en_to_hi"),
        ("val-en_to_hi.json", "en_to_hi"),
        ("train-hi_to_en.json", "hi_to_en"),
        ("val-hi_to_en.json", "hi_to_en")
    ]
    
    total_fixed = 0
    total_skipped = 0
    
    for filename, direction in datasets:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        fixed, skipped = fix_dataset(input_path, output_path, direction)
        total_fixed += fixed
        total_skipped += skipped
        
        print(f"Fixed {filename}: {fixed} entries, skipped {skipped} entries")
    
    print(f"\nTotal fixed: {total_fixed} entries")
    print(f"Total skipped: {total_skipped} entries")
    print(f"Fixed datasets saved to {output_dir}")

if __name__ == "__main__":
    main() 
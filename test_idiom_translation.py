#!/usr/bin/env python3
"""
Simple script to test idiom translations with the current model.
"""

import sys
import os
import csv
from src.translator import HindiEnglishTranslator

# Sample idioms to test
sample_idioms = [
    ("It's raining cats and dogs", "मूसलाधार बारिश हो रही है"),
    ("Break a leg", "शुभकामनाएं"),
    ("Piece of cake", "बहुत आसान काम"),
    ("Hit the nail on the head", "सही बात कहना"),
    ("Spill the beans", "राज़ खोल देना"),
    ("When pigs fly", "कभी नहीं")
]

def main():
    print("Testing idiom translations with the current model...\n")
    
    # Initialize translator
    translator = HindiEnglishTranslator()
    
    print("{:<30} | {:<30} | {:<30}".format("English Idiom", "Expected Hindi", "Model Translation"))
    print("-" * 95)
    
    for english, expected_hindi in sample_idioms:
        # Translate from English to Hindi
        translated = translator.translate_en_to_hi(english)
        
        print("{:<30} | {:<30} | {:<30}".format(
            english if len(english) < 30 else english[:27] + "...", 
            expected_hindi if len(expected_hindi) < 30 else expected_hindi[:27] + "...",
            translated if len(translated) < 30 else translated[:27] + "..."
        ))
    
    print("\nNow testing Hindi to English...\n")
    
    print("{:<30} | {:<30} | {:<30}".format("Hindi Idiom", "Expected English", "Model Translation"))
    print("-" * 95)
    
    for expected_english, hindi in sample_idioms:
        # Translate from Hindi to English
        translated = translator.translate_hi_to_en(hindi)
        
        print("{:<30} | {:<30} | {:<30}".format(
            hindi if len(hindi) < 30 else hindi[:27] + "...", 
            expected_english if len(expected_english) < 30 else expected_english[:27] + "...",
            translated if len(translated) < 30 else translated[:27] + "..."
        ))

if __name__ == "__main__":
    main() 
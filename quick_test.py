#!/usr/bin/env python3
"""
Quick test script to verify that the Hindi-English Translator is working correctly.
Run this after setting up the project to ensure everything is working.
"""
import logging
from src.translator import HindiEnglishTranslator

# Configure logging to show [testing] logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def run_test():
    print("\n===== Hindi-English Translator Quick Test =====\n")
    
    # Create translator
    print("Initializing translator...")
    translator = HindiEnglishTranslator()
    
    # Test English to Hindi translation
    english_text = "Hello, how are you doing today?"
    print(f"\nTranslating English → Hindi: '{english_text}'")
    
    try:
        hindi_translation = translator.translate_en_to_hi(english_text)
        print(f"Translation: '{hindi_translation}'")
        print("✓ English to Hindi translation successful!")
    except Exception as e:
        print(f"✗ English to Hindi translation failed: {str(e)}")
    
    # Test Hindi to English translation
    hindi_text = "नमस्ते, आज आप कैसे हैं?"
    print(f"\nTranslating Hindi → English: '{hindi_text}'")
    
    try:
        english_translation = translator.translate_hi_to_en(hindi_text)
        print(f"Translation: '{english_translation}'")
        print("✓ Hindi to English translation successful!")
    except Exception as e:
        print(f"✗ Hindi to English translation failed: {str(e)}")
    
    # Test idiom translation
    idiom = "It's raining cats and dogs."
    print(f"\nTranslating idiom: '{idiom}'")
    
    try:
        idiom_translation = translator.translate_en_to_hi(idiom)
        print(f"Translation: '{idiom_translation}'")
        print("✓ Idiom translation successful!")
    except Exception as e:
        print(f"✗ Idiom translation failed: {str(e)}")
    
    print("\n===== Test Complete =====")
    print("\nIf all tests passed, your setup is working correctly!")

if __name__ == "__main__":
    run_test() 
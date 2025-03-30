#!/usr/bin/env python3
import argparse
from src.translator import HindiEnglishTranslator

def main():
    """Run the Hindi-English Translator from the command line."""
    parser = argparse.ArgumentParser(description="Hindi-English Translator CLI")
    parser.add_argument("--text", "-t", required=True, help="Text to translate")
    parser.add_argument("--direction", "-d", choices=["en_to_hi", "hi_to_en"], default="en_to_hi",
                       help="Translation direction (en_to_hi or hi_to_en)")
    parser.add_argument("--model", "-m", help="Path to a fine-tuned model (optional)")
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = HindiEnglishTranslator(model_name=args.model) if args.model else HindiEnglishTranslator()
    
    # Translate text
    if args.direction == "en_to_hi":
        translation = translator.translate_en_to_hi(args.text)
    else:  # hi_to_en
        translation = translator.translate_hi_to_en(args.text)
    
    # Print results
    print("\nOriginal text:")
    print(args.text)
    print("\nTranslated text:")
    print(translation)

if __name__ == "__main__":
    main() 
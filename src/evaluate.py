import pandas as pd
import logging
import evaluate
from src.translator import HindiEnglishTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_translator(test_data_path, model_path=None, direction="en_to_hi"):
    """
    Evaluate the translator model on a test dataset.
    
    Args:
        test_data_path (str): Path to the test data CSV
        model_path (str): Path to the fine-tuned model (if None, use the default model)
        direction (str): Translation direction ('en_to_hi' or 'hi_to_en')
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info(f"[testing] Evaluating translator for {direction} direction")
    
    # Load the test data
    try:
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data from {test_data_path} with {len(test_df)} examples")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None
    
    # Initialize the translator
    translator = HindiEnglishTranslator(model_name=model_path) if model_path else HindiEnglishTranslator()
    
    # Initialize metrics
    bleu_metric = evaluate.load("sacrebleu")
    
    # Determine source and target columns based on direction
    if direction == "en_to_hi":
        source_col = "english"
        target_col = "hindi"
        translate_fn = translator.translate_en_to_hi
    elif direction == "hi_to_en":
        source_col = "hindi"
        target_col = "english"
        translate_fn = translator.translate_hi_to_en
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'en_to_hi' or 'hi_to_en'")
    
    # Perform translations
    sources = test_df[source_col].tolist()
    references = test_df[target_col].tolist()
    
    logger.info(f"[testing] Translating {len(sources)} examples")
    translations = []
    for i, source in enumerate(sources):
        translation = translate_fn(source)
        translations.append(translation)
        
        # Log some examples
        if i < 5 or i % 10 == 0:
            logger.info(f"[testing] Source: {source}")
            logger.info(f"[testing] Translation: {translation}")
            logger.info(f"[testing] Reference: {references[i]}")
    
    # Calculate BLEU score
    bleu_result = bleu_metric.compute(predictions=translations, references=[[ref] for ref in references])
    
    # Calculate exact match rate for idioms (simplified approach)
    idiom_indices = [i for i, s in enumerate(test_df[source_col]) if "idiom" in s.lower()]
    if idiom_indices:
        idiom_matches = sum(1 for i in idiom_indices if translations[i] == references[i])
        idiom_match_rate = idiom_matches / len(idiom_indices) if idiom_indices else 0
    else:
        idiom_match_rate = None
    
    # Create evaluation results
    results = {
        "bleu": bleu_result["score"],
        "idiom_match_rate": idiom_match_rate,
        "num_examples": len(sources)
    }
    
    logger.info(f"Evaluation results: {results}")
    return results

def compare_translations(text, direction="en_to_hi", custom_model_path=None):
    """
    Compare translations between the base model and our fine-tuned model.
    
    Args:
        text (str): Text to translate
        direction (str): Translation direction ('en_to_hi' or 'hi_to_en')
        custom_model_path (str): Path to the fine-tuned model
        
    Returns:
        dict: Dictionary containing base and fine-tuned translations
    """
    logger.info(f"[testing] Comparing translations for: {text}")
    
    # Initialize the base translator
    base_translator = HindiEnglishTranslator()
    
    # Initialize the custom translator if a path is provided
    custom_translator = None
    if custom_model_path:
        custom_translator = HindiEnglishTranslator(model_name=custom_model_path)
    
    # Get translations
    if direction == "en_to_hi":
        base_translation = base_translator.translate_en_to_hi(text)
        custom_translation = custom_translator.translate_en_to_hi(text) if custom_translator else None
    elif direction == "hi_to_en":
        base_translation = base_translator.translate_hi_to_en(text)
        custom_translation = custom_translator.translate_hi_to_en(text) if custom_translator else None
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'en_to_hi' or 'hi_to_en'")
    
    # Print translations
    logger.info(f"Base model translation: {base_translation}")
    if custom_translator:
        logger.info(f"Fine-tuned model translation: {custom_translation}")
    
    return {
        "text": text,
        "base_translation": base_translation,
        "custom_translation": custom_translation
    }

if __name__ == "__main__":
    # Example usage
    # evaluate_translator("data/test_data.csv", direction="en_to_hi")
    # compare_translations("It's raining cats and dogs", direction="en_to_hi")
    pass 
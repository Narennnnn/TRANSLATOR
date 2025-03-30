import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class HindiEnglishTranslator:
    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt", device=None):
        """
        Initialize the Hindi-English translator with a pre-trained model.
        
        Args:
            model_name (str): Pre-trained model name
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[testing] Using device: {self.device}")
        
        logger.info(f"Loading model: {model_name}")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        logger.info("Model loaded successfully")
        
        # Set default language codes
        self.hindi_code = "hi_IN"
        self.english_code = "en_XX"
    
    def translate_en_to_hi(self, text, max_length=1024):
        """
        Translate English text to Hindi.
        
        Args:
            text (str): English text to translate
            max_length (int): Maximum length of the generated translation
            
        Returns:
            str: Translated Hindi text
        """
        logger.info("[testing] Translating from English to Hindi")
        self.tokenizer.src_lang = self.english_code
        
        # Encode the English text
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.hindi_code],
            max_length=max_length
        )
        
        # Decode the translation
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logger.info(f"[testing] Original: {text}")
        logger.info(f"[testing] Translation: {translation}")
        
        return translation
    
    def translate_hi_to_en(self, text, max_length=1024):
        """
        Translate Hindi text to English.
        
        Args:
            text (str): Hindi text to translate
            max_length (int): Maximum length of the generated translation
            
        Returns:
            str: Translated English text
        """
        logger.info("[testing] Translating from Hindi to English")
        self.tokenizer.src_lang = self.hindi_code
        
        # Encode the Hindi text
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.english_code],
            max_length=max_length
        )
        
        # Decode the translation
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logger.info(f"[testing] Original: {text}")
        logger.info(f"[testing] Translation: {translation}")
        
        return translation
        
    def fine_tune(self, train_dataset, val_dataset, output_dir, epochs=3, batch_size=8):
        """
        Fine-tune the model on a custom dataset.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir (str): Directory to save the fine-tuned model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        # This method will be implemented later
        pass 
import os
import logging
import numpy as np
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TranslationTrainer:
    def __init__(
        self,
        model_name="facebook/mbart-large-50-many-to-many-mmt",
        output_dir="./models",
        device=None
    ):
        """
        Initialize the translation trainer.
        
        Args:
            model_name (str): Pre-trained model name
            output_dir (str): Directory to save the fine-tuned model
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"[testing] Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        logger.info("Model loaded successfully")
        
        # Set default language codes
        self.hindi_code = "hi_IN"
        self.english_code = "en_XX"
        
        # Initialize metric
        self.metric = evaluate.load("sacrebleu")

    def compute_metrics(self, pred):
        """
        Compute BLEU score for model evaluation.
        
        Args:
            pred: Prediction output from the model
            
        Returns:
            dict: Dictionary containing the BLEU score
        """
        logger.info("[testing] Computing metrics")
        
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        # Replace -100 with the pad token ID
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        
        # Decode predictions and references
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100  # Temporary fix for BLEU calculation
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        
        # Calculate BLEU score
        bleu = self.metric.compute(predictions=pred_str, references=[[ref] for ref in label_str])
        
        return {"bleu": bleu["score"]}

    def fine_tune(
        self,
        train_dataset,
        eval_dataset,
        direction="en_to_hi",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=500,
        max_grad_norm=1.0,
        weight_decay=0.01
    ):
        """
        Fine-tune the translation model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            direction (str): Translation direction ('en_to_hi' or 'hi_to_en')
            num_train_epochs (int): Number of training epochs
            per_device_train_batch_size (int): Batch size for training
            per_device_eval_batch_size (int): Batch size for evaluation
            gradient_accumulation_steps (int): Number of gradient accumulation steps
            learning_rate (float): Learning rate
            warmup_steps (int): Number of warmup steps
            max_grad_norm (float): Maximum gradient norm
            weight_decay (float): Weight decay
            
        Returns:
            Seq2SeqTrainer: Trained model trainer
        """
        # Set source and target language based on direction
        if direction == "en_to_hi":
            src_lang = self.english_code
            tgt_lang = self.hindi_code
        elif direction == "hi_to_en":
            src_lang = self.hindi_code
            tgt_lang = self.english_code
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'en_to_hi' or 'hi_to_en'")
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=100,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=5,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Start training
        logger.info(f"[testing] Starting fine-tuning for {direction} translation")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model(os.path.join(self.output_dir, f"final-{direction}"))
        logger.info(f"Model saved to {os.path.join(self.output_dir, f'final-{direction}')}")
        
        return trainer
    
    def evaluate_on_test(self, trainer, test_dataset):
        """
        Evaluate the fine-tuned model on the test dataset.
        
        Args:
            trainer: Trained Seq2SeqTrainer
            test_dataset: Test dataset
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("[testing] Evaluating model on test dataset")
        results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {results}")
        return results
    
    def translate_examples(self, examples, direction="en_to_hi", max_length=128):
        """
        Translate a batch of examples using the fine-tuned model.
        
        Args:
            examples (list): List of text examples to translate
            direction (str): Translation direction ('en_to_hi' or 'hi_to_en')
            max_length (int): Maximum length of the generated translation
            
        Returns:
            list: List of translated text examples
        """
        # Set source and target language based on direction
        if direction == "en_to_hi":
            src_lang = self.english_code
            tgt_lang = self.hindi_code
            source_key = "english"
        elif direction == "hi_to_en":
            src_lang = self.hindi_code
            tgt_lang = self.english_code
            source_key = "hindi"
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'en_to_hi' or 'hi_to_en'")
        
        # Set source language for tokenizer
        self.tokenizer.src_lang = src_lang
        
        # Tokenize input text
        inputs = self.tokenizer(examples, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        
        # Generate translations
        translated_ids = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_length,
            num_beams=5
        )
        
        # Decode translations
        translations = self.tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        
        return translations 
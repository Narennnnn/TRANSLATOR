#!/usr/bin/env python3
"""
Script for fine-tuning the mBART model on idioms for Google Colab.
This script is optimized for GPU usage on Colab and includes data loading,
processing, and fine-tuning.
"""

import argparse
import json
import logging
import os
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_jsonl_dataset(file_path):
    """Load dataset from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_metrics(eval_preds):
    """Compute BLEU score for evaluation."""
    metric = evaluate.load("sacrebleu")
    
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in the labels with pad token id
    labels = labels.reshape(-1)
    labels = [[l for l in labels if l != -100]]
    
    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU score calculation
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    return result

def fine_tune_model(model_name, train_data_path, val_data_path, output_dir, direction, args):
    """Fine-tune the translation model with GPU acceleration."""
    global tokenizer  # Make tokenizer available to compute_metrics
    
    # Determine source and target languages
    if direction == "en_to_hi":
        src_lang = "en_XX"
        tgt_lang = "hi_IN"
    else:  # hi_to_en
        src_lang = "hi_IN"
        tgt_lang = "en_XX"
    
    # Load tokenizer
    logger.info(f"[testing] Loading tokenizer from {model_name}")
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
    
    # Load model
    logger.info(f"[testing] Loading model from {model_name}")
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"[testing] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("[testing] GPU not available, using CPU")
    
    # Load datasets
    logger.info(f"[testing] Loading datasets from {train_data_path} and {val_data_path}")
    train_data = load_jsonl_dataset(train_data_path)
    val_data = load_jsonl_dataset(val_data_path)
    
    # Use smaller subsets for testing if needed
    if args.subset_size > 0:
        logger.info(f"[testing] Using subset of {args.subset_size} examples for training")
        train_data = train_data[:args.subset_size]
        val_data = val_data[:max(5, int(args.subset_size * 0.2))]
    
    # Convert to HF Datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Define preprocessing function
    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["target"]
        
        # Apply the tokenizer
        model_inputs = tokenizer(
            inputs, 
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Preprocess datasets
    logger.info("[testing] Preprocessing datasets")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=args.batch_size,
    )
    
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=args.batch_size,
    )
    
    # Set up data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments optimized for Colab GPU
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        fp16=torch.cuda.is_available(),  # Enable mixed precision training for GPU
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        report_to="tensorboard"
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Fine-tune model
    logger.info(f"[testing] Starting fine-tuning for {direction}")
    trainer.train()
    
    # Save model
    logger.info(f"[testing] Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"[testing] Fine-tuning complete for {direction}")
    return trainer

def test_translations(model_dir, direction, test_samples):
    """Test the fine-tuned model on some samples."""
    logger.info(f"[testing] Testing fine-tuned model from {model_dir}")
    
    # Determine source and target languages
    if direction == "en_to_hi":
        src_lang = "en_XX"
        tgt_lang = "hi_IN"
    else:  # hi_to_en
        src_lang = "hi_IN"
        tgt_lang = "en_XX"
    
    # Load model and tokenizer
    tokenizer = MBart50Tokenizer.from_pretrained(model_dir, src_lang=src_lang, tgt_lang=tgt_lang)
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Test translations
    logger.info("Sample translations:")
    for sample in test_samples:
        inputs = tokenizer(sample, return_tensors="pt", max_length=128, padding=True, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        generated_tokens = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            num_return_sequences=1,
            temperature=1.0
        )
        
        translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print(f"Input: {sample}")
        print(f"Translation: {translations[0]}\n")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune mBART model on idioms (Colab GPU optimized)")
    parser.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt", 
                        help="Base model to fine-tune")
    parser.add_argument("--data-dir", default="data/fine_tuning_fixed", 
                        help="Directory containing processed training data")
    parser.add_argument("--output-dir", default="models/fine_tuned", 
                        help="Directory to save fine-tuned models")
    parser.add_argument("--direction", choices=["en_to_hi", "hi_to_en", "both"], default="en_to_hi", 
                        help="Translation direction to fine-tune")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size for training (can be larger on Colab GPU)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, 
                        help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=128, 
                        help="Maximum token length for inputs/outputs")
    parser.add_argument("--subset-size", type=int, default=0, 
                        help="Number of examples to use for training (0 to use all)")
    parser.add_argument("--test", action="store_true",
                        help="Test the model after training")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fine-tune models based on direction
    if args.direction in ["en_to_hi", "both"]:
        en_to_hi_output = os.path.join(args.output_dir, "final-en_to_hi")
        os.makedirs(en_to_hi_output, exist_ok=True)
        
        train_data_path = os.path.join(args.data_dir, "train-en_to_hi.json")
        val_data_path = os.path.join(args.data_dir, "val-en_to_hi.json")
        
        trainer = fine_tune_model(
            args.model_name,
            train_data_path,
            val_data_path,
            en_to_hi_output,
            "en_to_hi",
            args
        )
        
        # Test some translations
        if args.test:
            test_samples = [
                "It's raining cats and dogs",
                "Break a leg",
                "A penny for your thoughts",
                "Out of the blue"
            ]
            test_translations(en_to_hi_output, "en_to_hi", test_samples)
    
    if args.direction in ["hi_to_en", "both"]:
        hi_to_en_output = os.path.join(args.output_dir, "final-hi_to_en")
        os.makedirs(hi_to_en_output, exist_ok=True)
        
        train_data_path = os.path.join(args.data_dir, "train-hi_to_en.json")
        val_data_path = os.path.join(args.data_dir, "val-hi_to_en.json")
        
        trainer = fine_tune_model(
            args.model_name,
            train_data_path,
            val_data_path,
            hi_to_en_output,
            "hi_to_en",
            args
        )
        
        # Test some translations
        if args.test:
            test_samples = [
                "मूसलाधार बारिश हो रही है",
                "शुभकामनाएं",
                "अचानक",
                "नई शुरुआत"
            ]
            test_translations(hi_to_en_output, "hi_to_en", test_samples)
    
    logger.info("[testing] All fine-tuning tasks completed!")

if __name__ == "__main__":
    main() 
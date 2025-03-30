# Hindi-English Translator
# This is a Python script version of a notebook that can be run in Google Colab
# To use in Colab, copy and paste each cell into a Colab notebook

# Cell 1: Setup
# --------------
# Clone the repository
!git clone https://github.com/your-username/Hindi-English-Translator.git
# Change to the project directory
%cd Hindi-English-Translator
# Install dependencies
!pip install -r requirements.txt

# Cell 2: Create directories
# --------------------------
!mkdir -p data models

# Cell 3: Create Sample Dataset
# -----------------------------
from src.create_sample_dataset import create_sample_dataset

# Create and display the sample dataset
df = create_sample_dataset()
df.head(10)

# Cell 4: Basic Translation
# -------------------------
from src.translator import HindiEnglishTranslator

# Initialize the translator
translator = HindiEnglishTranslator()

# Test English to Hindi translation
english_text = "Hello, how are you doing today?"
hindi_translation = translator.translate_en_to_hi(english_text)
print(f"English: {english_text}")
print(f"Hindi: {hindi_translation}")

# Test Hindi to English translation
hindi_text = "नमस्ते, आज आप कैसे हैं?"
english_translation = translator.translate_hi_to_en(hindi_text)
print(f"\nHindi: {hindi_text}")
print(f"English: {english_translation}")

# Cell 5: Testing with Idioms
# ---------------------------
# Test with English idioms
english_idioms = [
    "It's raining cats and dogs.",
    "Break a leg on your performance!",
    "That's a piece of cake.",
    "I'm feeling blue today.",
    "He kicked the bucket last year."
]

print("English Idioms to Hindi Translation:")
for idiom in english_idioms:
    hindi = translator.translate_en_to_hi(idiom)
    print(f"English: {idiom}")
    print(f"Hindi: {hindi}")
    print("-" * 50)

# Test with Hindi idioms
hindi_idioms = [
    "आंखों में धूल झोंकना",
    "नाक में दम करना",
    "दाल में कुछ काला होना",
    "आसमान सिर पर उठाना",
    "अंधे के हाथ बटेर लगना"
]

print("\nHindi Idioms to English Translation:")
for idiom in hindi_idioms:
    english = translator.translate_hi_to_en(idiom)
    print(f"Hindi: {idiom}")
    print(f"English: {english}")
    print("-" * 50)

# Cell 6: Data Processing
# -----------------------
from src.data_processor import DataProcessor

# Initialize the data processor
data_processor = DataProcessor()

# Load the sample dataset
df = data_processor.load_data_from_csv('data/sample_dataset.csv')

# Create train, validation, and test datasets
dataset_dict = data_processor.create_datasets(df)

# Print dataset statistics
print(f"Train dataset size: {len(dataset_dict['train'])}")
print(f"Validation dataset size: {len(dataset_dict['validation'])}")
print(f"Test dataset size: {len(dataset_dict['test'])}")

# Cell 7: Check GPU Availability
# -----------------------------
# Check if GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 8: Fine-tuning (Optional)
# ------------------------------
from src.trainer import TranslationTrainer

# Prepare datasets for English to Hindi translation
en_to_hi_datasets = data_processor.prepare_datasets(dataset_dict, direction="en_to_hi")

# Initialize the trainer
trainer = TranslationTrainer(output_dir="./models")

# Fine-tune the model (uncomment to run)
# Note: This can take a long time depending on your hardware
# Also, for a proper fine-tuning, you need a larger dataset

# trainer.fine_tune(
#     en_to_hi_datasets["train"],
#     en_to_hi_datasets["validation"],
#     direction="en_to_hi",
#     num_train_epochs=1,  # Reduced for demonstration
#     per_device_train_batch_size=4  # Reduced for memory constraints
# )

# Cell 9: Downloading Additional Datasets (Optional)
# -------------------------------------------------
# Download the IIT Bombay dataset (uncomment to run)
# !python -c "from setup_colab import download_dataset; download_dataset('iitb')"

# Cell 10: Web Interface (Optional)
# --------------------------------
# Run the web interface (uncomment to run)
# For Google Colab, you need to install flask-ngrok first
# !pip install flask-ngrok
# from src.web_interface import create_app
# from flask_ngrok import run_with_ngrok
# app = create_app()
# run_with_ngrok(app)
# app.run()

# Cell 11: Comparative Evaluation
# ------------------------------
from src.evaluate import compare_translations

# Path to fine-tuned model (if available)
fine_tuned_model_path = None  # Change this if you've fine-tuned a model

# Compare translations for some challenging examples
examples = [
    "It's raining cats and dogs outside.",
    "Break a leg at your performance tonight!",
    "Finding a good job in this economy is like looking for a needle in a haystack.",
    "The new manager is all bark and no bite.",
    "When pigs fly, that's when he'll apologize."
]

for example in examples:
    result = compare_translations(example, direction="en_to_hi", custom_model_path=fine_tuned_model_path)
    print("-" * 70) 
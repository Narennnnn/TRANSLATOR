# Hindi-English Translator

A context-aware Hindi-English translator that properly handles idioms and phrases in both languages. Built using the Transformer-based neural machine translation approach.

## Features

- Bidirectional translation between Hindi and English
- Special focus on handling idioms and contextual phrases
- Fine-tuning capabilities for improving translation quality
- Web interface for easy interaction
- Evaluation tools for testing translation quality
- Local development in Cursor IDE

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/Narennnnn/TRANSLATOR.git
cd TRANSLATOR
```

2. Run the local setup script:
```bash
python local_setup.py
```

This will:
- Install all required dependencies
- Set up the package in development mode
- Download necessary resources
- Create a sample dataset

## Project Structure

```
├── data/                  # Data directory for datasets
├── models/                # Directory to store fine-tuned models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── translator.py      # Main translator module
│   ├── data_processor.py  # Data processing utilities
│   ├── trainer.py         # Model training utilities
│   ├── evaluate.py        # Evaluation tools
│   ├── web_interface.py   # Web interface
│   ├── main.py            # Command-line interface
│   └── create_sample_dataset.py # Sample dataset creation
├── local_setup.py         # Local setup script
├── run_translator.py      # Script to run the translator
├── run_web_interface.py   # Script to run the web interface
├── train_model.py         # Script to train the model
├── setup.py               # Package setup script
└── requirements.txt       # Python dependencies
```

## Usage

### Command Line Interface

Translate text using the command line:
```bash
python run_translator.py --text "Hello, how are you?" --direction en_to_hi
```

```bash
python run_translator.py --text "नमस्ते, आप कैसे हैं?" --direction hi_to_en
```

### Web Interface

Run the web interface:
```bash
python run_web_interface.py
```

Then open your browser and go to http://localhost:5000

### Train a Model

Train the translator on your data:
```bash
python train_model.py --data data/your_dataset.csv --direction en_to_hi --epochs 3
```

### Python API

```python
from src.translator import HindiEnglishTranslator

# Initialize translator
translator = HindiEnglishTranslator()

# Translate English to Hindi
hindi_text = translator.translate_en_to_hi("Hello, how are you?")

# Translate Hindi to English
english_text = translator.translate_hi_to_en("नमस्ते, आप कैसे हैं?")
```

## Datasets

We provide a small sample dataset for testing. For better results, download larger datasets:

```bash
python local_setup.py iitb  # Download IIT Bombay Hindi-English Corpus
python local_setup.py wmt   # Download WMT Hindi-English dataset
```

## Development in Cursor IDE

This project is optimized for development in Cursor IDE:

1. Open the project folder in Cursor
2. Run the local setup script:
   ```bash
   python local_setup.py
   ```
3. Use the provided scripts to run the translator, web interface, or train models
4. Edit the source code in the `src` directory
5. Make changes and run the scripts to test your changes

## Fine-tuning

To fine-tune the model with your own data:

1. Prepare a CSV file with 'english' and 'hindi' columns
2. Run the training script:
```bash
python train_model.py --data data/your_dataset.csv --direction en_to_hi --epochs 3
```

Or use the Python API:
```python
from src.data_processor import DataProcessor
from src.trainer import TranslationTrainer

# Load and process data
data_processor = DataProcessor()
df = data_processor.load_data_from_csv('data/your_dataset.csv')
datasets = data_processor.create_datasets(df)
prepared_datasets = data_processor.prepare_datasets(datasets, direction="en_to_hi")

# Train the model
trainer = TranslationTrainer(output_dir="./models")
trainer.fine_tune(
    prepared_datasets["train"],
    prepared_datasets["validation"],
    direction="en_to_hi",
    num_train_epochs=3
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [mBART](https://arxiv.org/abs/2001.08210)
- [IIT Bombay Hindi-English Corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) 

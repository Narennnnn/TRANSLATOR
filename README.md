# Hindi-English Translator

A context-aware Hindi-English translator that properly handles idioms and phrases in both languages. Built using the Transformer-based neural machine translation approach.

## Features

- Bidirectional translation between Hindi and English
- Special focus on handling idioms and contextual phrases
- Fine-tuning capabilities for improving translation quality
- Web interface for easy interaction
- Evaluation tools for testing translation quality
- Google Colab compatibility

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/Narennnnn/TRANSLATOR
cd TRANSLATOR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Google Colab Installation

Run our setup script in Google Colab:
```python
!pip install git+https://github.com/Narennnnn/TRANSLATOR.git
!python -c "from setup_colab import setup_colab; setup_colab()"
```

## Project Structure

```
├── data/                  # Data directory for datasets
├── models/                # Directory to store fine-tuned models
├── notebooks/             # Jupyter notebooks
│   └── Hindi_English_Translator_colab.py  # Google Colab script
├── src/                   # Source code
│   ├── translator.py      # Main translator module
│   ├── data_processor.py  # Data processing utilities
│   ├── trainer.py         # Model training utilities
│   ├── evaluate.py        # Evaluation tools
│   ├── web_interface.py   # Web interface
│   ├── main.py            # Command-line interface
│   └── create_sample_dataset.py # Sample dataset creation
├── setup_colab.py         # Setup script for Google Colab
└── requirements.txt       # Python dependencies
```

## Usage

### Command Line Interface

Translate text from English to Hindi:
```bash
python src/main.py translate "Hello, how are you?" --direction en_to_hi
```

Translate text from Hindi to English:
```bash
python src/main.py translate "नमस्ते, आप कैसे हैं?" --direction hi_to_en
```

Fine-tune the model:
```bash
python src/main.py train --data data/your_dataset.csv --direction en_to_hi --epochs 3
```

### Web Interface

Run the web interface:
```bash
python -c "from src.web_interface import create_app; app = create_app(); app.run(debug=True)"
```

Then open your browser and go to http://localhost:5000

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

## Google Colab

You can run this translator in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy and paste from `notebooks/Hindi_English_Translator_colab.py`

## Datasets

We provide a small sample dataset for testing. For better results, download larger datasets:

```python
from setup_colab import download_dataset

# Download IIT Bombay Hindi-English Corpus
download_dataset('iitb')

# Download WMT Hindi-English dataset
download_dataset('wmt')
```

## Fine-tuning

To fine-tune the model with your own data:

1. Prepare a CSV file with 'english' and 'hindi' columns
2. Run the training script:
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

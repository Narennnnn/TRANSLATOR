# Hindi-English Translator with Idiom Support

A context-aware Hindi-English translator that properly handles idioms and phrases in both languages. Built using the Transformer-based neural machine translation approach with fine-tuning for idiomatic expressions.

[![Demo Video](https://cdn.loom.com/sessions/thumbnails/b0d60637e22147eabd5803dab2b7f0e6-with-play.gif)](https://www.loom.com/share/b0d60637e22147eabd5803dab2b7f0e6?sid=ee0f6ce4-6b09-47ea-8144-1319e71ec052)

## Features

- Bidirectional translation between Hindi and English
- Special focus on handling idioms and contextual phrases
- Fine-tuning capabilities for improving translation quality
- Web interface for easy interaction
- Based on mBART-50 multilingual translation model
- Optimized for both local and cloud-based (Google Colab) environments

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/Narennnnn/TRANSLATOR.git
cd TRANSLATOR
git clone https://github.com/Narennnnn/TRANSLATOR.git
cd TRANSLATOR
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv fresh_venv
source fresh_venv/bin/activate  # On Windows: fresh_venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the setup script:
```bash
python local_setup.py
```

## Project Structure

```
├── data/                  # Data directory for datasets
│   └── fine_tuning_fixed/ # Fixed datasets for fine-tuning idioms
├── models/                # Directory to store fine-tuned models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── translator.py      # Main translator module
│   ├── web_interface.py   # Web interface
├── finetune_colab.py      # Script for fine-tuning on Google Colab
├── run_translator.py      # Script to run the translator
├── run_web_interface.py   # Script to run the web interface
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

To use a fine-tuned model with the web interface:
```bash
python run_web_interface.py --models-dir models/fine_tuned
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

## Fine-tuning for Idioms

The main contribution of this project is the ability to fine-tune the model to correctly translate idiomatic expressions.

### Fine-tuning on Google Colab

For the best experience and faster training, we recommend using Google Colab with GPU acceleration:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SRbubS-0kI4m7OgtqeX4nHDN5ugrOCD6?usp=sharing)

The Colab notebook includes all necessary steps to:
1. Set up the environment
2. Load and prepare the idiomatic expression datasets
3. Fine-tune the mBART model
4. Test the fine-tuned model
5. Save the model for deployment

### Local Fine-tuning

For local fine-tuning:

1. Make sure you have a GPU with sufficient memory
2. Prepare your idiomatic expression dataset in JSONL format
3. Run the fine-tuning script:

```bash
python finetune_colab.py --direction en_to_hi --epochs 5 --batch-size 4 --data-dir data/fine_tuning_fixed
```

## Using Fine-tuned Models

After fine-tuning, you can use the models with the web interface:

```bash
python run_web_interface.py --models-dir models/fine_tuned
```

This will load your fine-tuned models for both Hindi to English and English to Hindi translation.

## Research Papers

This project is based on the following research papers:

1. [mBART: Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) - Liu et al., 2020
2. [Beyond Literal Translation: Cross-Lingual Idiom Processing with Neural Networks](https://aclanthology.org/2023.findings-acl.426/) - Liu et al., 2023
3. [Neural Machine Translation Does Not Translate Idioms Well: Exposing Idiom Processing Deficiencies and A Path Forward](https://aclanthology.org/2023.acl-long.764/) - Tayyar Madabushi et al., 2023
4. [Understanding Idioms: A Cross-Lingual Study on Questions and Answers Involving Idioms](https://aclanthology.org/2020.coling-main.348/) - Tanabe et al., 2020
5. [A Survey on Transfer Learning for Neural Machine Translation](https://www.mdpi.com/2076-3417/11/19/9092) - Bhattacharyya et al., 2021

### Additional Related Research

6. [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401) - Tang et al., 2020 - Introduces mBART-50 and multilingual fine-tuning strategies
7. [Phrase-Based & Neural Unsupervised Machine Translation](https://aclanthology.org/D18-1549/) - Lample et al., 2018 - Pioneering work on unsupervised NMT techniques
8. [Automatically Identifying Idioms in Hindi](https://aclanthology.org/W16-6330/) - Manwani et al., 2016 - Specific focus on Hindi idiom identification
9. [Low-Resource Neural Machine Translation for Southern African Languages](https://arxiv.org/abs/2203.15987) - Abbott and Martinus, 2022 - Insights on low-resource language translation applicable to Hindi
10. [Indic-Transformers: An Analysis of Transformer Language Models for Indian Languages](https://arxiv.org/abs/2011.02323) - Jain et al., 2020 - Analysis of transformer models for Indian languages
11. [English-Hindi Neural Machine Translation and Low Resource South Indian Languages](https://arxiv.org/abs/2109.14814) - Philip and Namboodiri, 2021 - Specific focus on English-Hindi translation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [mBART-50 Many-to-Many Multilingual Translation](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
- [IIT Bombay Hindi-English Corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) 

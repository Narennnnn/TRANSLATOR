# Hindi-English Idiom Translation Fine-Tuning

This guide explains how to fine-tune the Hindi-English Translator specifically for idiomatic expressions. Idioms are phrases that have a figurative meaning different from their literal meaning, which makes them particularly challenging for machine translation.

## Prerequisites

- Python 3.8+
- Virtual environment with requirements installed
- Base Hindi-English Translator project set up

## Quick Start

1. Install additional requirements for fine-tuning:
   ```
   pip install -r requirements-finetune.txt
   ```

2. Prepare a CSV file of idioms:
   ```
   data/custom_idioms/idioms.csv
   ```
   Format: `english,hindi` with one idiom pair per line

3. Process the idioms data:
   ```
   python src/prepare_idiom_dataset.py
   ```

4. Fine-tune the model:
   ```
   python src/finetune_model.py --direction both --epochs 3 --batch-size 4
   ```

5. Evaluate the fine-tuned model:
   ```
   python src/evaluate_idioms.py --direction en_to_hi --fine-tuned-model models/fine_tuned/final-en_to_hi
   ```

## Detailed Instructions

### 1. Creating an Idiom Dataset

Create a CSV file with English idioms and their Hindi translations. The CSV should have two columns:
- `english`: The English idiom
- `hindi`: The Hindi equivalent

Example:
```csv
english,hindi
"It's raining cats and dogs","मूसलाधार बारिश हो रही है"
"Break a leg","शुभकामनाएं"
```

Save this file as `data/custom_idioms/idioms.csv`.

### 2. Preparing the Dataset

Run the dataset preparation script:

```
python src/prepare_idiom_dataset.py --csv data/custom_idioms/idioms.csv
```

This will:
- Read the idioms from the CSV file
- Create training and validation sets for both translation directions
- Format the data for mBART fine-tuning
- Save the processed datasets in JSONL format

### 3. Fine-tuning the Model

Run the fine-tuning script:

```
python src/finetune_model.py --direction both --epochs 3
```

Options:
- `--direction`: Choose `en_to_hi`, `hi_to_en`, or `both`
- `--epochs`: Number of training epochs (3-5 recommended)
- `--batch-size`: Batch size (adjust based on your GPU memory)
- `--learning-rate`: Learning rate (default: 5e-5)

Fine-tuning will:
- Load the base mBART model
- Train it on the idiom dataset
- Save the fine-tuned model in `models/fine_tuned/`

### 4. Evaluating the Model

Test how well the fine-tuned model translates idioms:

```
python src/evaluate_idioms.py --direction en_to_hi --fine-tuned-model models/fine_tuned/final-en_to_hi
```

This will:
- Compare translations from the base model and fine-tuned model
- Save the results in a CSV file for analysis

### 5. Using the Fine-tuned Model

After fine-tuning, update your translation code to use the fine-tuned model:

```python
from src.translator import HindiEnglishTranslator

# Use the fine-tuned model
translator = HindiEnglishTranslator(model_name="models/fine_tuned/final-en_to_hi")

# Translate an idiom
result = translator.translate_en_to_hi("It's raining cats and dogs")
print(result)  # Should output "मूसलाधार बारिश हो रही है"
```

## Fine-tuning Tips

1. **Dataset Size**: Aim for at least 100 idioms for reasonable fine-tuning results
2. **Training Time**: Fine-tuning takes significant time and computational resources
3. **Evaluation**: Compare the base model and fine-tuned model translations to measure improvement
4. **GPU Requirements**: Fine-tuning works best with a GPU
5. **Reduced Batch Size**: If running out of memory, reduce batch size (e.g., `--batch-size 2`)

## Troubleshooting

- **Memory Errors**: Reduce batch size or max sequence length
- **Training Time**: Reduce number of epochs for faster training
- **Overfitting**: If fine-tuned model performs poorly on new idioms, increase dataset size
- **Poor Performance**: Ensure idiom translations are accurate in your dataset 
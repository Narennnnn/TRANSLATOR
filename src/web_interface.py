from flask import Flask, render_template, request, jsonify
import logging
import os
from src.translator import HindiEnglishTranslator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize translators
en_to_hi_translator = None
hi_to_en_translator = None

def load_translators(custom_model_dir=None):
    """
    Load translator models.
    
    Args:
        custom_model_dir (str): Directory containing fine-tuned models
    """
    global en_to_hi_translator, hi_to_en_translator
    
    logger.info("[testing] Loading translator models")
    
    # Check if fine-tuned models exist
    en_to_hi_model_path = os.path.join(custom_model_dir, "final-en_to_hi") if custom_model_dir else None
    hi_to_en_model_path = os.path.join(custom_model_dir, "final-hi_to_en") if custom_model_dir else None
    
    # Load English to Hindi translator
    en_to_hi_model = en_to_hi_model_path if (en_to_hi_model_path and os.path.exists(en_to_hi_model_path)) else None
    en_to_hi_translator = HindiEnglishTranslator(model_name=en_to_hi_model) if en_to_hi_model else HindiEnglishTranslator()
    logger.info(f"Loaded English to Hindi translator from {en_to_hi_model if en_to_hi_model else 'default model'}")
    
    # Load Hindi to English translator
    hi_to_en_model = hi_to_en_model_path if (hi_to_en_model_path and os.path.exists(hi_to_en_model_path)) else None
    hi_to_en_translator = HindiEnglishTranslator(model_name=hi_to_en_model) if hi_to_en_model else HindiEnglishTranslator()
    logger.info(f"Loaded Hindi to English translator from {hi_to_en_model if hi_to_en_model else 'default model'}")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate text using the appropriate translator.
    
    Request JSON:
        text (str): Text to translate
        direction (str): Translation direction ('en_to_hi' or 'hi_to_en')
        
    Returns:
        JSON with translated text
    """
    try:
        # Get request data
        data = request.get_json()
        text = data.get('text', '')
        direction = data.get('direction', 'en_to_hi')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"[testing] Translating: {text} ({direction})")
        
        # Translate using the appropriate translator
        if direction == 'en_to_hi':
            translated_text = en_to_hi_translator.translate_en_to_hi(text)
        elif direction == 'hi_to_en':
            translated_text = hi_to_en_translator.translate_hi_to_en(text)
        else:
            return jsonify({'error': 'Invalid direction'}), 400
        
        return jsonify({'translated_text': translated_text})
    
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_app(custom_model_dir=None):
    """
    Create and configure the Flask app.
    
    Args:
        custom_model_dir (str): Directory containing fine-tuned models
        
    Returns:
        Flask app
    """
    # Create static and templates directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Create templates/index.html
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hindi-English Translator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .translator {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .text-area {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: none;
            font-size: 16px;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .direction-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .direction-toggle button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
        }
        .translate-btn {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .error {
            color: red;
            text-align: center;
            margin-top: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hindi-English Translator</h1>
        <div class="translator">
            <div class="text-area">
                <h3>Input Text</h3>
                <textarea id="input-text" placeholder="Enter text to translate..."></textarea>
            </div>
            
            <div class="controls">
                <div class="direction-toggle">
                    <label>Direction:</label>
                    <select id="direction">
                        <option value="en_to_hi">English → Hindi</option>
                        <option value="hi_to_en">Hindi → English</option>
                    </select>
                </div>
                
                <button class="translate-btn" id="translate-btn">Translate</button>
            </div>
            
            <div class="loading" id="loading">
                <p>Translating...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="text-area">
                <h3>Translation</h3>
                <textarea id="output-text" placeholder="Translation will appear here..." readonly></textarea>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const translateBtn = document.getElementById('translate-btn');
            const inputText = document.getElementById('input-text');
            const outputText = document.getElementById('output-text');
            const directionSelect = document.getElementById('direction');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            
            translateBtn.addEventListener('click', function() {
                const text = inputText.value.trim();
                if (!text) {
                    showError('Please enter text to translate');
                    return;
                }
                
                const direction = directionSelect.value;
                
                // Show loading
                loading.style.display = 'block';
                error.style.display = 'none';
                outputText.value = '';
                
                // Send translation request
                fetch('/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text,
                        direction: direction
                    })
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    
                    if (data.error) {
                        showError(data.error);
                    } else {
                        outputText.value = data.translated_text;
                    }
                })
                .catch(err => {
                    loading.style.display = 'none';
                    showError('Error occurred during translation');
                    console.error(err);
                });
            });
            
            function showError(message) {
                error.textContent = message;
                error.style.display = 'block';
            }
        });
    </script>
</body>
</html>
        """)
    
    # Load translators
    load_translators(custom_model_dir)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 
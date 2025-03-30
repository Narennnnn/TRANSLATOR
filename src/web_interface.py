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

# Create Flask app with template_folder set to the absolute path
template_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

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
    logger.info("Rendering index page")
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
    # Make sure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Load translators
    load_translators(custom_model_dir)
    
    logger.info(f"Flask app created with template folder: {template_dir}")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 
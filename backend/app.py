from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import multiprocessing
import os
from typing import Dict, Optional
from functools import lru_cache
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration class for default settings."""
    DEFAULT_SOURCE_LANG: str = "en"  # Default source language
    DEFAULT_TARGET_LANG: str = "es"  # Default target language
    MAX_TEXT_LENGTH: int = 1000      # Maximum allowed text length for translation
    MODEL_BASE_PATH: str = "Helsinki-NLP/opus-mt"  # Base path for translation models

config = Config()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all web browsers to access endpoints

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Function to determine the unique key for rate limiting
    default_limits=["100 per day", "10 per minute"]  # Default rate limits
)

# Cache for translation models
@lru_cache(maxsize=5)
# Replace your get_translator and translate sections with this:

def get_translation_result(text: str, source_lang: str, target_lang: str) -> str:
    """Loads the pipeline, translates immediately, and frees memory right after."""
    language_models = {
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
        ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
        ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    }

    model_name = language_models.get((source_lang, target_lang))
    if not model_name:
        raise ValueError(f"Unsupported language pair: {source_lang}-{target_lang}")

    logger.info(f"Loading model on-demand: {model_name}")
    
    # Initialize pipeline locally so it can be garbage collected
    translator_pipeline = pipeline("translation", model=model_name)
    result = translator_pipeline(text, max_length=400)
    translated_text = result[0]['translation_text']
    
    # Clean up memory explicitly to prevent Render Free Tier crashes
    del translator_pipeline
    import gc
    gc.collect()
    
    return translated_text

@app.route('/translate', methods=['POST'])
@limiter.limit("10 per minute")
def translate():
    try:
        request_data = request.json
        input_text = request_data.get("text", "")
        source_language = request_data.get("source_language", config.DEFAULT_SOURCE_LANG)
        target_language = request_data.get("target_language", config.DEFAULT_TARGET_LANG)

        # Validate input
        validation_error = validate_input(input_text, source_language, target_language)
        if validation_error:
            return jsonify(validation_error[0]), validation_error[1]

        # Translate and immediately flush RAM
        translated_text = get_translation_result(input_text, source_language, target_language)

        logger.info(f"Successfully translated text from {source_language} to {target_language}")
        return jsonify({
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language
        })

    except ValueError as validation_exception:
        logger.error(f"Validation error: {str(validation_exception)}")
        return jsonify({"error": str(validation_exception)}), 400
    except Exception as general_exception:
        import traceback
        traceback.print_exc()
        logger.error(f"Translation error: {str(general_exception)}")
        return jsonify({"error": str(general_exception)}), 500

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Support for Windows multiprocessing
    
    # Grab Render's dynamic port environment variable, or fall back to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    
    # Binding to '0.0.0.0' exposes the app onto the public cloud network interface
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import multiprocessing
import os
from typing import Dict, Optional
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dataclasses import dataclass
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    DEFAULT_SOURCE_LANG: str = "en"
    DEFAULT_TARGET_LANG: str = "es"
    MAX_TEXT_LENGTH: int = 500  # Lowered slightly to save RAM bursts

config = Config()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

def validate_input(text: str, source_language: str, target_language: str) -> Optional[Dict]:
    if not text:
        return {"error": "No text provided"}, 400
    if len(text) > config.MAX_TEXT_LENGTH:
        return {"error": f"Text exceeds maximum length of {config.MAX_TEXT_LENGTH}"}, 400
    if source_language == target_language:
        return {"error": "Source and target languages must be different"}, 400
    return None

def get_translation_result(text: str, source_lang: str, target_lang: str) -> str:
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

    logger.info(f"On-demand memory-restricted loading: {model_name}")
    
    # Load pipeline with explicit low_cpu_mem_usage flag to stay under 512MB
    translator_pipeline = pipeline(
        "translation", 
        model=model_name,
        low_cpu_mem_usage=True
    )
    
    result = translator_pipeline(text, max_length=150)
    translated_text = result[0]['translation_text']
    
    # Aggressive RAM cleanup right after generation
    del translator_pipeline
    gc.collect()
    
    return translated_text

@app.route('/translate', methods=['POST'])
@limiter.limit("10 per minute")
def translate():
    try:
        request_data = request.json or {}
        input_text = request_data.get("text", "")
        source_language = request_data.get("source_language", config.DEFAULT_SOURCE_LANG)
        target_language = request_data.get("target_language", config.DEFAULT_TARGET_LANG)

        validation_error = validate_input(input_text, source_language, target_language)
        if validation_error:
            return jsonify(validation_error[0]), validation_error[1]

        translated_text = get_translation_result(input_text, source_language, target_language)
        return jsonify({
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language
        })

    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as err:
        logger.error(f"Execution Error: {str(err)}")
        return jsonify({"error": "Model loading memory timeout. Please try again in 5 seconds."}), 500

if __name__ == '__main__':
    multiprocessing.freeze_support()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

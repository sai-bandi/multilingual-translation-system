from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Explicitly open CORS configurations for smooth Vercel data pipeline handshakes
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json or {}
        text = data.get("text", "")
        src = data.get("source_language", "en")
        tgt = data.get("target_language", "es")
        
        if not text or not text.strip():
            return jsonify({"error": "No text provided"}), 400

        if src == tgt:
            return jsonify({"error": "Source and target languages must be different"}), 400

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        
        logger.info(f"Shipping text to Hugging Face Cloud Engine: {model_name}")
        response = requests.post(api_url, json=payload)
        result = response.json()
        
        # --- ROBUST TYPE PARSING STRATEGY FOR ALL HUGGING FACE RESPONSES ---
        if isinstance(result, list) and len(result) > 0 and 'translation_text' in result[0]:
            translated_text = result[0]['translation_text']
            
        elif isinstance(result, dict):
            if 'translation_text' in result:
                translated_text = result['translation_text']
            elif 'error' in result:
                # Catch the exact sleep/warmup message cleanly without breaking the response format
                error_msg = result.get('error', '')
                logger.warning(f"Hugging Face Warmup Active: {error_msg}")
                return jsonify({"error": f"Cloud engine warming up. Please click Translate again in 5 seconds."}), 503
            else:
                translated_text = str(result)
        else:
            translated_text = str(result)

        return jsonify({
            "translated_text": translated_text, 
            "source_language": src, 
            "target_language": tgt
        })
        
    except Exception as e:
        logger.error(f"Global Pipeline Connection Failure: {str(e)}")
        return jsonify({"error": "Translation pipeline temporarily busy. Click Translate again."}), 500

if __name__ == '__main__':
    multiprocessing.freeze_support()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

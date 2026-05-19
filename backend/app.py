from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import multiprocessing

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json or {}
        text = data.get("text", "")
        src = data.get("source_language", "en")
        tgt = data.get("target_language", "es")
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # FORCES cloud engine to wait until warm instead of dropping timeout!
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True} 
        }
        
        response = requests.post(api_url, json=payload)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0 and 'translation_text' in result[0]:
            translated_text = result[0]['translation_text']
        else:
            translated_text = "Model loading infrastructure processing setup. Click translate again."

        return jsonify({
            "translated_text": translated_text, 
            "source_language": src, 
            "target_language": tgt
        })
        
    except Exception as e:
        return jsonify({"error": "Translation cluster delay. Please try again."}), 500

if __name__ == '__main__':
    multiprocessing.freeze_support()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

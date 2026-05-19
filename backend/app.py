from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Allow cross-origin requests from Vercel cleanly
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
            
        if src == tgt:
            return jsonify({"error": "Source and target languages must be different"}), 400

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        # Connect directly to Hugging Face cloud servers (0% RAM impact on Render!)
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        logger.info(f"Routing request to Hugging Face API Serverless Interface: {model_name}")
        response = requests.post(api_url, json={"inputs": text})
        
        if response.status_code != 200:
            return jsonify({"error": "Cloud model engine warming up. Click translate again in 5 seconds."}), 500
            
        result = response.json()
        translated_text = result[0]['translation_text']
        
        return jsonify({
            "translated_text": translated_text, 
            "source_language": src, 
            "target_language": tgt
        })
        
    except Exception as e:
        logger.error(f"API Connection Error: {str(e)}")
        return jsonify({"error": "Translation delay. Please try again."}), 500

if __name__ == '__main__':
    multiprocessing.freeze_support()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

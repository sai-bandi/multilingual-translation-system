from flask import Flask, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator
import multiprocessing
import os

app = Flask(__name__)
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

        # Instant free cloud translator engine - No sleep cycles, No rate limits
        translated_text = GoogleTranslator(source=src, target=tgt).translate(text)

        return jsonify({
            "translated_text": translated_text, 
            "source_language": src, 
            "target_language": tgt
        })
        
    except Exception as e:
        return jsonify({"error": "Translation service temporary delay. Please try again."}), 500

if __name__ == '__main__':
    multiprocessing.freeze_support()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

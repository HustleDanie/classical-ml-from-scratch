"""
Model Deployment API -- Flask REST API
=======================================
Serves the trained text classification model as a REST API.

Setup:
  pip install flask

Usage:
  python deploy_api.py

Test with curl:
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"The new GPU from NVIDIA has amazing compute performance\"}"

Test with Python:
  import requests
  resp = requests.post('http://localhost:5000/predict',
                       json={'text': 'The new GPU has great performance'})
  print(resp.json())
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load model + vectorizer (saved by advanced_ml_pipeline.py)
MODEL_DIR = Path(__file__).parent
model = joblib.load(MODEL_DIR / "model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'text-classifier-v1'})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether text is tech-related or not."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text']
    threshold = data.get('threshold', 0.5)  # Allow custom threshold!

    # Vectorize + predict
    features = vectorizer.transform([text])
    probability = model.predict_proba(features)[0]
    prediction = int(probability[1] >= threshold)

    return jsonify({
        'prediction': prediction,
        'label': 'tech' if prediction == 1 else 'non-tech',
        'confidence': float(max(probability)),
        'probability_tech': float(probability[1]),
        'threshold_used': threshold,
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple texts."""
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing "texts" field'}), 400

    texts = data['texts']
    threshold = data.get('threshold', 0.5)

    features = vectorizer.transform(texts)
    probabilities = model.predict_proba(features)
    predictions = (probabilities[:, 1] >= threshold).astype(int)

    results = []
    for text, pred, prob in zip(texts, predictions, probabilities):
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': int(pred),
            'label': 'tech' if pred else 'non-tech',
            'probability_tech': float(prob[1]),
        })
    return jsonify({'predictions': results, 'count': len(results)})


if __name__ == '__main__':
    print("Starting Text Classification API...")
    print("Endpoints:")
    print("  GET  /health         -- Health check")
    print("  POST /predict        -- Single prediction")
    print("  POST /predict_batch  -- Batch predictions")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)

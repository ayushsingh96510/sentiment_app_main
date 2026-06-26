# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pickle
from pathlib import Path
import os
import json
import numpy as np

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = Path('model/sentiment_model.pkl')

app = Flask(__name__)

# Store analysis history in memory (in production, use a database)
analysis_history = []

# ======================
# LOAD MODEL
# ======================
def load_model():
    """Load the trained sentiment model"""
    if not MODEL_PATH.exists():
        raise RuntimeError(f'Model not found at {MODEL_PATH}. Run `python train_model.py` first.')
    
    try:
        # Try loading with joblib first (if saved with joblib)
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully with joblib")
        return model
    except:
        try:
            # Fallback to pickle (if saved with pickle)
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully with pickle")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

model = load_model()

# ======================
# SENTIMENT MAPPING
# ======================
# Map numeric labels to human-readable format
SENTIMENT_MAP = {
    -1: {'label': 'Negative', 'emoji': '😞', 'color': '#ff4444'},
    0: {'label': 'Neutral', 'emoji': '😐', 'color': '#ffbb33'},
    1: {'label': 'Positive', 'emoji': '😊', 'color': '#00C851'}
}

def predict_sentiment(text: str):
    """
    Predict sentiment for a given text.
    Returns: (label_string, confidence_score, emoji, color)
    """
    # Clean the text (matching training preprocessing)
    from train_model import preprocess_text
    cleaned_text = preprocess_text(text)
    
    # Get prediction
    pred = model.predict([cleaned_text])[0]
    
    # Get confidence scores
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba([cleaned_text])[0]
        # Get the class corresponding to the prediction
        # For sklearn pipelines, get the classifier from the last step
        classifier = model.named_steps[list(model.named_steps.keys())[-1]]
        classes = classifier.classes_
        
        # Map probabilities to classes
        class_probs = dict(zip(classes, probs))
        confidence = float(class_probs.get(pred, max(probs)))
    else:
        confidence = 0.0
    
    # Map numeric label to readable format
    sentiment_info = SENTIMENT_MAP.get(pred, {'label': 'Unknown', 'emoji': '❓', 'color': '#888888'})
    
    return {
        'numeric_label': int(pred),
        'label': sentiment_info['label'],
        'emoji': sentiment_info['emoji'],
        'color': sentiment_info['color'],
        'confidence': round(confidence, 4)
    }

# ======================
# ROUTES
# ======================
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request from the web form"""
    text = request.form.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get prediction
    result = predict_sentiment(text)
    
    # Add to history
    analysis_history.append({
        'text': text,
        'label': result['label'],
        'numeric_label': result['numeric_label'],
        'confidence': result['confidence'],
        'emoji': result['emoji'],
        'color': result['color']
    })
    
    # Keep only last 50 analyses in history
    if len(analysis_history) > 50:
        analysis_history.pop(0)
    
    # Return JSON for AJAX handling
    return jsonify({
        'text': text,
        'label': result['label'],
        'numeric_label': result['numeric_label'],
        'confidence': result['confidence'],
        'emoji': result['emoji'],
        'color': result['color']
    })

@app.route('/get_history', methods=['GET'])
def get_history():
    """Return analysis history for the chart"""
    return jsonify(analysis_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all analysis history"""
    global analysis_history
    analysis_history.clear()
    return jsonify({'success': True, 'message': 'History cleared successfully'})

@app.route('/get_chart_data', methods=['GET'])
def get_chart_data():
    """Return data for the sentiment distribution chart"""
    sentiment_counts = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    for analysis in analysis_history:
        label_lower = analysis['label'].lower()
        if 'positive' in label_lower:
            sentiment_counts['positive'] += 1
        elif 'negative' in label_lower:
            sentiment_counts['negative'] += 1
        else:
            sentiment_counts['neutral'] += 1
    
    return jsonify(sentiment_counts)

# ======================
# JSON API (for external apps)
# ======================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for prediction"""
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = predict_sentiment(text)
    return jsonify({
        'label': result['label'],
        'numeric_label': result['numeric_label'],
        'confidence': result['confidence'],
        'emoji': result['emoji']
    })

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """Batch prediction for multiple texts"""
    data = request.get_json(force=True)
    texts = data.get('texts', [])
    
    if not texts or not isinstance(texts, list):
        return jsonify({'error': 'Please provide a list of texts'}), 400
    
    results = []
    for text in texts:
        if text.strip():
            result = predict_sentiment(text.strip())
            results.append({
                'text': text,
                'label': result['label'],
                'confidence': result['confidence']
            })
    
    return jsonify({
        'total': len(results),
        'results': results
    })

# ======================
# ERROR HANDLING
# ======================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ======================
# RUN THE APP
# ======================
if __name__ == '__main__':
    # Use port 5000 by default
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask app on port {port}")
    print(f"📊 Model loaded: {MODEL_PATH}")
    print(f"📍 Visit: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
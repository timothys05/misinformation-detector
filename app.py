"""
app.py - Flask web application for misinformation detection
"""

from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'models/classifier.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("❌ Model files not found. Please train the model first.")
    model = None
    vectorizer = None

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on submitted text"""
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get text from form
        text = request.form.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Please enter some text'}), 400
        
        # Transform text using vectorizer
        text_tfidf = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        # Get confidence score
        confidence = max(probability) * 100
        
        # Determine result
        result = "LIKELY FAKE" if prediction == 1 else "LIKELY REAL"
        
        return render_template('result.html', 
                             text=text,
                             prediction=result,
                             confidence=f"{confidence:.1f}",
                             prob_real=f"{probability[0]*100:.1f}",
                             prob_fake=f"{probability[1]*100:.1f}")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Transform and predict
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'label': 'fake' if prediction == 1 else 'real',
            'confidence': float(max(probability)),
            'probabilities': {
                'real': float(probability[0]),
                'fake': float(probability[1])
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
import os
import re
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open('model/fake_review_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def lexical_analysis(text):
    words = text.split()
    exclamations = text.count('!')
    uppers = sum(1 for c in text if c.isupper())
    total_chars = max(len(text), 1)
    
    superlatives = ['best', 'greatest', 'miraculous', 'unbelievable', 'magical', 'perfection', 'flawless', 'incredible', 'a+++++']
    superlative_count = sum(1 for w in words if w.lower().strip('!?,.') in superlatives)
    
    return {
        'word_count': len(words),
        'exclamation_count': exclamations,
        'uppercase_ratio': round((uppers / total_chars) * 100, 1),
        'superlative_count': superlative_count
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() if request.is_json else request.form
    review_text = data.get('review', '').strip()
    
    if not review_text:
        return jsonify({'error': 'Please enter a review to analyze.'}), 400

    # ML Inference
    X_vec = vectorizer.transform([review_text])
    prob_fake = float(model.predict_proba(X_vec)[0][1])
    prob_genuine = float(model.predict_proba(X_vec)[0][0])
    
    # Heuristic adjustment for exaggerated punctuation/superlatives
    stats = lexical_analysis(review_text)
    heuristic_penalty = min(0.3, (stats['exclamation_count'] * 0.05) + (stats['superlative_count'] * 0.1))
    adjusted_prob_fake = min(0.99, max(0.01, prob_fake + heuristic_penalty))
    adjusted_prob_genuine = 1.0 - adjusted_prob_fake

    is_fake = adjusted_prob_fake >= 0.5
    verdict = "Deceptive / Fake Review" if is_fake else "Authentic / Genuine Review"
    confidence = round((adjusted_prob_fake if is_fake else adjusted_prob_genuine) * 100, 1)

    indicators = []
    if stats['exclamation_count'] >= 3:
        indicators.append("Excessive exclamation marks indicate synthetic enthusiasm.")
    if stats['superlative_count'] >= 2:
        indicators.append("Repetitive high-salience superlatives ('greatest', 'best ever', 'perfection').")
    if stats['uppercase_ratio'] > 15:
        indicators.append("High capital letter ratio indicating aggressive sales pitch.")
    if stats['word_count'] < 10 and is_fake:
        indicators.append("Extremely brief without functional or technical product details.")
    if not is_fake:
        indicators.append("Balanced vocabulary with specific experiential details and moderate sentiment.")

    return jsonify({
        'verdict': verdict,
        'is_fake': is_fake,
        'confidence': confidence,
        'prob_fake': round(adjusted_prob_fake * 100, 1),
        'prob_genuine': round(adjusted_prob_genuine * 100, 1),
        'stats': stats,
        'indicators': indicators
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

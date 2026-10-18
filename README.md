# Fake Product Review Detection System

Machine Learning system utilizing Natural Language Processing (NLP) to detect astroturfed, bot-generated, and incentivized fake online product reviews.

## Model Details
- **Architecture**: TF-IDF Feature Extraction + Logistic Regression Classifier.
- **Pre-trained Weights**: Pre-packaged under `model/fake_review_model.pkl` and `model/vectorizer.pkl`. Zero student training required.
- **Key Metrics**: Analyzes lexical diversity, uppercase density, superlative frequency, and emotional polarity.

## Quick Start
```bash
# 1. Activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Flask web application
python app.py
```
Open `http://127.0.0.1:5000` to run live review detection.

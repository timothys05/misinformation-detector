# Misinformation Detector

A machine learning web application that classifies text as likely real or fake news.

## Video Link

[INSERT YOUR YOUTUBE VIDEO LINK HERE]

## Project Overview and Purpose

This project uses machine learning to detect potential misinformation in news articles. It demonstrates text classification using TF-IDF feature extraction and logistic regression to predict whether content is likely authentic or fake.

## Installation and Setup Instructions

### Prerequisites
- Python 3.11 or 3.12
- pip package manager

### Setup Steps
```bash
# 1. Clone repository
git clone [YOUR-REPO-URL]
cd misinformation-detector

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
# Place in data/Fake.csv

# 5. Train model
python code/train_model.py

# 6. Run application
python code/app.py

# 7. Open browser
# Go to http://localhost:5000
```

## How to Run the Program

**Web Interface:**
```bash
python code/app.py
```
Then open http://localhost:5000 in your browser.

**Run Tests:**
```bash
python tests/test_predictions.py
```

## Reproduce Results

1. Use the Kaggle dataset: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
2. Train with `random_state=42` (already set in code)
3. 80/20 train/test split (already set in code)

## Technologies Used

- Python 3.11+
- scikit-learn 1.7.2 (TF-IDF, Logistic Regression)
- Flask 3.1.2 (web framework)
- pandas 2.3.3 (data handling)
- numpy 2.3.5
- joblib 1.5.2 (model saving)

## Author and Contribution

**Author:** Timothy Suh
**Course:** CS 2104  
**Semester:** Fall 2024

**Solo project - all work completed by Timothy Suh**

- Implemented ML pipeline with TF-IDF and logistic regression
- Built Flask web application
- Created test suite
- Used Claude AI for code assistance and debugging
- Dataset from Kaggle

**Total Time:** 11 hours
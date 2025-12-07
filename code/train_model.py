"""
train_model.py - Train the misinformation classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_and_prepare_data(fake_path, real_path=None):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    
    # Load fake news
    df_fake = pd.read_csv(fake_path)
    df_fake['label'] = 1  # 1 = fake
    
    # Load real news if available
    if real_path and os.path.exists(real_path):
        df_real = pd.read_csv(real_path)
        df_real['label'] = 0  # 0 = real
        df = pd.concat([df_fake, df_real], ignore_index=True)
        print(f"Combined dataset - Fake: {len(df_fake)}, Real: {len(df_real)}")
    else:
        print("Warning: Only fake news dataset loaded. Cannot train real vs fake classifier.")
        print("Please provide a real news dataset or use subject classification instead.")
        df = df_fake
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    df = df.dropna()
    
    # Combine title and text
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'] + ' ' + df['text']
    elif 'text' in df.columns:
        df['content'] = df['text']
    elif 'title' in df.columns:
        df['content'] = df['title']
    else:
        raise ValueError("No text column found in dataset")
    
    if 'label' not in df.columns:
        raise ValueError("No label column found. Please provide both fake and real news datasets.")
    
    return df['content'], df['label']

def train_classifier(X, y):
    """Train the classification model"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create TF-IDF vectorizer
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train logistic regression
    print("Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    train_pred = model.predict(X_train_tfidf)
    test_pred = model.predict(X_test_tfidf)
    
    print(f"Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    
    return model, vectorizer

def save_model(model, vectorizer, model_dir='models'):
    """Save the trained model and vectorizer"""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'classifier.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\n✓ Model saved to {model_path}")
    print(f"✓ Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    FAKE_DATA_PATH = 'data/Fake.csv'
    REAL_DATA_PATH = 'data/True.csv'  # or 'data/Real.csv'
    
    try:
        # Load data
        X, y = load_and_prepare_data(FAKE_DATA_PATH, REAL_DATA_PATH)
        
        # Train model
        model, vectorizer = train_classifier(X, y)
        
        # Save model
        save_model(model, vectorizer)
        
        print("\nTraining complete!")
        
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}")
        print("Please download the dataset and place it in the data/ folder")
    except Exception as e:
        print(f"Error: {e}")
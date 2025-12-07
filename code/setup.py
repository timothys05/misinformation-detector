"""
Misinformation Detector - Setup Script
Run this to create project structure and requirements file
"""

import os

# Create project directories
directories = [
    'data',
    'models',
    'templates',
    'static',
    'notebooks'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Created {directory}/ directory")

# Create requirements.txt
requirements = """flask==3.0.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)
print("✓ Created requirements.txt")

# Create a simple README
readme = """# Misinformation Detector

A machine learning project to classify text as real or fake news.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download dataset and place in `data/` folder
3. Run `train_model.py` to train the classifier
4. Run `app.py` to start the web interface

## Project Structure
- `data/` - Dataset files
- `models/` - Saved ML models
- `templates/` - HTML templates for Flask
- `static/` - CSS and JavaScript files
- `notebooks/` - Jupyter notebooks for exploration
"""

with open('README.md', 'w') as f:
    f.write(readme)
print("✓ Created README.md")

print("\n Project structure created successfully!")
print("\nNext steps:")
print("1. Run: pip install -r requirements.txt")
print("2. Download dataset from Kaggle")
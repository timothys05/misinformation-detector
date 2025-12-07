"""
test_predictions.py - Unit tests for misinformation detector
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import joblib
import pytest

# Load model and vectorizer
MODEL_PATH = 'models/classifier.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("Model files not found. Train the model first.")
    sys.exit(1)

def predict_text(text):
    """Helper function to make predictions"""
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    return {
        'prediction': int(prediction),
        'label': 'fake' if prediction == 1 else 'real',
        'confidence': float(max(probability)),
        'prob_real': float(probability[0]),
        'prob_fake': float(probability[1])
    }

# Test cases
class TestMisinformationDetector:
    
    def test_empty_string(self):
        """Test handling of empty string"""
        result = predict_text("")
        assert 'prediction' in result
        assert 'confidence' in result
        print("Empty string test passed")
    
    def test_very_short_text(self):
        """Test very short input"""
        result = predict_text("Hello")
        assert result['confidence'] >= 0.0 and result['confidence'] <= 1.0
        print("Short text test passed")
    
    def test_real_news_example(self):
        """Test with typical real news text"""
        text = """The Federal Reserve announced today that it will maintain 
        interest rates at their current levels, citing ongoing economic stability 
        and controlled inflation. The decision was made after careful review of 
        economic indicators."""
        
        result = predict_text(text)
        print(f"Real news example - Prediction: {result['label']}, Confidence: {result['confidence']:.2f}")
        assert result['prediction'] in [0, 1]
        print("Real news test passed")
    
    def test_fake_news_example(self):
        """Test with typical fake news text"""
        text = """BREAKING: Scientists discover that drinking coffee backwards 
        cures all diseases! Doctors hate this one weird trick! Share before 
        the government removes this information!"""
        
        result = predict_text(text)
        print(f"Fake news example - Prediction: {result['label']}, Confidence: {result['confidence']:.2f}")
        assert result['prediction'] in [0, 1]
        print("Fake news test passed")
    
    def test_special_characters(self):
        """Test handling of special characters"""
        text = "!!!??? This is a test with special characters @#$%^&*()"
        result = predict_text(text)
        assert 'prediction' in result
        print("Special characters test passed")
    
    def test_numbers_only(self):
        """Test handling of numeric input"""
        result = predict_text("123 456 789")
        assert result['confidence'] >= 0.0 and result['confidence'] <= 1.0
        print("Numbers test passed")
    
    def test_very_long_text(self):
        """Test handling of very long text"""
        long_text = "This is a test sentence. " * 200
        result = predict_text(long_text)
        assert 'prediction' in result
        print("Long text test passed")
    
    def test_probabilities_sum_to_one(self):
        """Test that probabilities sum to approximately 1"""
        text = "This is a test"
        result = predict_text(text)
        prob_sum = result['prob_real'] + result['prob_fake']
        assert abs(prob_sum - 1.0) < 0.01  # Allow small floating point error
        print(f"Probability sum test passed (sum={prob_sum:.4f})")
    
    def test_confidence_matches_max_probability(self):
        """Test that confidence equals max probability"""
        text = "Testing confidence calculation"
        result = predict_text(text)
        max_prob = max(result['prob_real'], result['prob_fake'])
        assert abs(result['confidence'] - max_prob) < 0.001
        print("Confidence calculation test passed")
    
    def test_multiple_predictions_consistency(self):
        """Test that same input gives same prediction"""
        text = "Consistency test text"
        result1 = predict_text(text)
        result2 = predict_text(text)
        assert result1['prediction'] == result2['prediction']
        assert abs(result1['confidence'] - result2['confidence']) < 0.001
        print("Consistency test passed")

def run_comprehensive_tests():
    """Run all tests and print results"""
    print("\n" + "="*60)
    print("MISINFORMATION DETECTOR - TEST SUITE")
    print("="*60 + "\n")
    
    test_suite = TestMisinformationDetector()
    tests = [
        test_suite.test_empty_string,
        test_suite.test_very_short_text,
        test_suite.test_real_news_example,
        test_suite.test_fake_news_example,
        test_suite.test_special_characters,
        test_suite.test_numbers_only,
        test_suite.test_very_long_text,
        test_suite.test_probabilities_sum_to_one,
        test_suite.test_confidence_matches_max_probability,
        test_suite.test_multiple_predictions_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
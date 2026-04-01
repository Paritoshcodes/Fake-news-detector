import pytest
from src.models.predict import predict_article
from sklearn.datasets import make_classification
import numpy as np

# Sample data for testing
def create_sample_data():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    return X, y

# Test for the predict_article function
def test_predict_article():
    # Assuming the model is already trained and loaded
    model = ...  # Load your trained model here
    vectorizer = ...  # Load your vectorizer here

    # Sample input for prediction
    title = "Sample News Title"
    text = "This is a sample news article content for testing."

    # Perform prediction
    prediction, confidence = predict_article(title, text, model, vectorizer)

    # Check if the prediction is one of the expected classes
    assert prediction in ['FAKE', 'REAL']
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 100

# Test for multiple predictions
def test_multiple_predictions():
    model = ...  # Load your trained model here
    vectorizer = ...  # Load your vectorizer here

    test_cases = [
        ("Breaking News: Something happened!", "Details about the event."),
        ("This is a fake news article.", "Content that is misleading."),
    ]

    for title, text in test_cases:
        prediction, confidence = predict_article(title, text, model, vectorizer)
        assert prediction in ['FAKE', 'REAL']
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100

# Run the tests
if __name__ == "__main__":
    pytest.main()
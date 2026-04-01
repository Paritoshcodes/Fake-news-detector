import pytest
from src.data.preprocess import preprocess_text

def test_preprocess_text_valid():
    input_text = "This is a sample text for testing."
    expected_output = "sampl text test"
    assert preprocess_text(input_text) == expected_output

def test_preprocess_text_empty():
    input_text = ""
    expected_output = ""
    assert preprocess_text(input_text) == expected_output

def test_preprocess_text_non_string():
    input_text = 12345
    expected_output = ""
    assert preprocess_text(input_text) == expected_output

def test_preprocess_text_with_url():
    input_text = "Check this link: http://example.com"
    expected_output = "check link"
    assert preprocess_text(input_text) == expected_output

def test_preprocess_text_with_special_characters():
    input_text = "Hello!!! This is a test @2023."
    expected_output = "hello test"
    assert preprocess_text(input_text) == expected_output
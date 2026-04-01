import pytest
from src.utils.web_scraper import extract_article_from_url

def test_extract_article_from_url_valid():
    url = "https://example.com/sample-article"
    title, text = extract_article_from_url(url)
    assert title is not None and title != ""
    assert text is not None and text != ""

def test_extract_article_from_url_invalid():
    url = "https://invalid-url.com"
    title, text = extract_article_from_url(url)
    assert title is None
    assert text is None

def test_extract_article_from_url_empty():
    url = ""
    title, text = extract_article_from_url(url)
    assert title is None
    assert text is None

def test_extract_article_from_url_no_title():
    url = "https://example.com/no-title-article"
    title, text = extract_article_from_url(url)
    assert title == ""
    assert text is not None and text != ""
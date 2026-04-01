import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def _normalize_url(url):
    if not isinstance(url, str):
        return None

    normalized = url.strip()
    if not normalized:
        return None

    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"

    parsed = urlparse(normalized)
    if not parsed.netloc:
        return None

    return normalized


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title.get("content").strip()

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    return ""


def _extract_main_text(soup: BeautifulSoup) -> str:
    candidates = [
        soup.find("article"),
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
        soup.find("div", attrs={"id": re.compile(r"content|article|main", re.I)}),
        soup.find("div", attrs={"class": re.compile(r"content|article|main|story", re.I)}),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        paragraphs = candidate.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if p.get_text(strip=True))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 280:
            return text

    fallback_paragraphs = soup.find_all("p")
    fallback_text = " ".join(
        p.get_text(" ", strip=True)
        for p in fallback_paragraphs
        if len(p.get_text(strip=True)) >= 35
    )
    return re.sub(r"\s+", " ", fallback_text).strip()


def extract_article_from_url(url):
    normalized_url = _normalize_url(url)
    if not normalized_url:
        return None, None

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(normalized_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses

        soup = BeautifulSoup(response.text, "html.parser")

        title = _extract_title(soup)
        text = _extract_main_text(soup)

        if not title and not text:
            return None, None

        if len(text) > 12000:
            text = text[:12000]

        return title, text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None, None

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
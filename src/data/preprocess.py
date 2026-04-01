import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


def _load_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception:
            return FALLBACK_STOPWORDS

stemmer = PorterStemmer()
stop_words = _load_stop_words()

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]

    return ' '.join(tokens)

def preprocess_dataframe(df):
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")

    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df[['cleaned_text', 'label']]
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from src.data.preprocess import preprocess_text
from src.utils.web_scraper import extract_article_from_url


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PROFILE_MODEL_PATHS = {
    "quick": ARTIFACTS_DIR / "fake_news_model_quick.joblib",
    "full": ARTIFACTS_DIR / "fake_news_model_full.joblib",
}
PROFILE_VECTORIZER_PATHS = {
    "quick": ARTIFACTS_DIR / "tfidf_vectorizer_quick.joblib",
    "full": ARTIFACTS_DIR / "tfidf_vectorizer_full.joblib",
}
LEGACY_MODEL_PATH = ARTIFACTS_DIR / "fake_news_model.joblib"
LEGACY_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
ACTIVE_PROFILE_PATH = ARTIFACTS_DIR / "active_profile.txt"

FAKE_HINTS = {
    "secret",
    "plot",
    "hoax",
    "miracle",
    "cure",
    "shocking",
    "exposed",
    "conspiracy",
    "clickbait",
    "outrage",
    "banned",
    "truth",
}

REAL_HINTS = {
    "report",
    "statement",
    "according",
    "official",
    "analysis",
    "source",
    "research",
    "data",
    "confirmed",
    "agency",
    "journal",
    "interview",
}


@lru_cache(maxsize=8)
def _load_artifact(path: str):
    return joblib.load(path)


def _profile_artifacts_exist(profile: str) -> bool:
    return PROFILE_MODEL_PATHS[profile].exists() and PROFILE_VECTORIZER_PATHS[profile].exists()


def _legacy_artifacts_exist() -> bool:
    return LEGACY_MODEL_PATH.exists() and LEGACY_VECTORIZER_PATH.exists()


def _read_active_profile() -> Optional[str]:
    if not ACTIVE_PROFILE_PATH.exists():
        return None

    try:
        profile = ACTIVE_PROFILE_PATH.read_text(encoding="utf-8").strip().lower()
        if profile in PROFILE_MODEL_PATHS:
            return profile
    except Exception:
        return None

    return None


def _normalize_label(prediction) -> str:
    if isinstance(prediction, (int, np.integer)):
        return "FAKE" if int(prediction) == 1 else "REAL"

    label = str(prediction).strip().upper()
    if label in {"1", "TRUE", "FAKE"}:
        return "FAKE"
    if label in {"0", "FALSE", "REAL", "LEGIT"}:
        return "REAL"
    return label


def _extract_confidence(model, vectorized_text) -> float:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized_text)
        return float(np.max(probs[0]) * 100)

    if hasattr(model, "decision_function"):
        raw_score = float(np.ravel(model.decision_function(vectorized_text))[0])
        score = 1.0 / (1.0 + np.exp(-abs(raw_score)))
        return float(score * 100)

    return 65.0


def _heuristic_predict(text: str) -> Tuple[str, float]:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    fake_hits = sum(1 for token in tokens if token in FAKE_HINTS)
    real_hits = sum(1 for token in tokens if token in REAL_HINTS)
    score = fake_hits - real_hits

    if score > 0:
        label = "FAKE"
    elif score < 0:
        label = "REAL"
    else:
        label = "REAL"

    base_confidence = 58 + min(30, abs(score) * 8)
    if len(tokens) < 15:
        base_confidence -= 6

    confidence = float(max(50, min(94, base_confidence)))
    return label, confidence


def get_active_profile() -> Optional[str]:
    configured = _read_active_profile()
    if configured and _profile_artifacts_exist(configured):
        return configured

    if _profile_artifacts_exist("full"):
        return "full"
    if _profile_artifacts_exist("quick"):
        return "quick"
    if _legacy_artifacts_exist():
        return "legacy"

    return None


def _load_default_bundle() -> Tuple[Optional[object], Optional[object]]:
    active_profile = get_active_profile()
    if not active_profile:
        return None, None

    if active_profile == "legacy":
        model_path = LEGACY_MODEL_PATH
        vectorizer_path = LEGACY_VECTORIZER_PATH
    else:
        model_path = PROFILE_MODEL_PATHS[active_profile]
        vectorizer_path = PROFILE_VECTORIZER_PATHS[active_profile]

    try:
        model = _load_artifact(str(model_path))
        vectorizer = _load_artifact(str(vectorizer_path))
        return model, vectorizer
    except Exception:
        return None, None


def model_artifacts_available(profile: Optional[str] = None) -> bool:
    if profile is None:
        return get_active_profile() is not None

    normalized = str(profile).strip().lower()
    if normalized in PROFILE_MODEL_PATHS:
        return _profile_artifacts_exist(normalized)
    if normalized == "legacy":
        return _legacy_artifacts_exist()

    return False


def get_prediction_backend() -> str:
    active_profile = get_active_profile()
    if not active_profile:
        return "Heuristic fallback"

    if active_profile == "legacy":
        return "WELFake trained model (legacy artifact)"

    return f"WELFake trained model ({active_profile} profile)"


def refresh_artifact_cache() -> None:
    _load_artifact.cache_clear()


def load_model(model_path: str):
    return _load_artifact(str(model_path))


def load_vectorizer(vectorizer_path: str):
    return _load_artifact(str(vectorizer_path))


def predict_text(text: str, model=None, vectorizer=None) -> Tuple[str, float]:
    if not isinstance(text, str) or not text.strip():
        return "UNKNOWN", 0.0

    model_obj = model
    vectorizer_obj = vectorizer
    if model_obj is None or vectorizer_obj is None:
        model_obj, vectorizer_obj = _load_default_bundle()

    if model_obj is not None and vectorizer_obj is not None:
        try:
            processed_text = preprocess_text(text)
            text_to_vectorize = processed_text if processed_text else text
            vectorized = vectorizer_obj.transform([text_to_vectorize])
            raw_prediction = model_obj.predict(vectorized)[0]
            confidence = _extract_confidence(model_obj, vectorized)
            return _normalize_label(raw_prediction), round(confidence, 2)
        except Exception:
            pass

    label, confidence = _heuristic_predict(text)
    return label, round(confidence, 2)


def predict_article(title: str, text: str, model=None, vectorizer=None) -> Tuple[str, float]:
    combined_text = f"{title or ''} {text or ''}".strip()
    return predict_text(combined_text, model=model, vectorizer=vectorizer)


def predict_url(url: str, model=None, vectorizer=None) -> Tuple[Optional[str], float]:
    title, text = extract_article_from_url(url)
    if not title and not text:
        return None, 0.0

    prediction, confidence = predict_article(title or "", text or "", model=model, vectorizer=vectorizer)
    return prediction, confidence


def load_and_predict(model_path: str, vectorizer_path: str, input_data: str, is_url: bool = False):
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    if is_url:
        return predict_url(input_data, model=model, vectorizer=vectorizer)

    return predict_text(input_data, model=model, vectorizer=vectorizer)
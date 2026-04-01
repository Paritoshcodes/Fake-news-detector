from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Literal, Optional
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_CACHE_DIR = ARTIFACTS_DIR / "datasets"
LEGACY_DATASET_PATH = ROOT_DIR / "WELFake_Dataset.csv"
DEFAULT_DATASET_PATH = DATASET_CACHE_DIR / "WELFake_Dataset.csv"
DEFAULT_DATASET_DRIVE_URL = "https://drive.google.com/file/d/13lcNYSvVfJhC5xl-84k5AcHvNVnKiI1T/view?usp=drive_link"
DATASET_DRIVE_URL_ENV = "WELFAKE_DATASET_URL"
TRAINING_PROFILES = ("quick", "full")

PROFILE_MAX_ROWS = {
    "quick": 30000,
    "full": None,
}

PROFILE_MODEL_PATHS = {
    "quick": ARTIFACTS_DIR / "fake_news_model_quick.joblib",
    "full": ARTIFACTS_DIR / "fake_news_model_full.joblib",
}

PROFILE_VECTORIZER_PATHS = {
    "quick": ARTIFACTS_DIR / "tfidf_vectorizer_quick.joblib",
    "full": ARTIFACTS_DIR / "tfidf_vectorizer_full.joblib",
}

PROFILE_METRICS_PATHS = {
    "quick": ARTIFACTS_DIR / "training_metrics_quick.json",
    "full": ARTIFACTS_DIR / "training_metrics_full.json",
}

ACTIVE_PROFILE_PATH = ARTIFACTS_DIR / "active_profile.txt"
ACTIVE_METRICS_PATH = ARTIFACTS_DIR / "training_metrics_active.json"

# Backward compatible alias used by existing pages.
DEFAULT_METRICS_PATH = ACTIVE_METRICS_PATH


def _validate_profile(profile: str) -> Literal["quick", "full"]:
    normalized = str(profile).strip().lower()
    if normalized not in TRAINING_PROFILES:
        raise ValueError(f"Unknown training profile: {profile}. Use 'quick' or 'full'.")
    return normalized  # type: ignore[return-value]


def get_welfake_dataset_source_url() -> str:
    configured = os.getenv(DATASET_DRIVE_URL_ENV, "").strip()
    return configured or DEFAULT_DATASET_DRIVE_URL


def _extract_google_drive_file_id(share_url: str) -> str:
    normalized = str(share_url).strip()
    if not normalized:
        raise ValueError("Google Drive dataset URL is empty.")

    # Common format: /file/d/<FILE_ID>/view
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", normalized)
    if match:
        return match.group(1)

    parsed = urlparse(normalized)
    query_values = parse_qs(parsed.query)
    query_id = query_values.get("id")
    if query_id and query_id[0]:
        return query_id[0]

    raise ValueError(
        "Could not parse Google Drive file id from dataset URL. "
        "Expected a link like 'https://drive.google.com/file/d/<id>/view'."
    )


def _response_is_html(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    content_disposition = response.headers.get("Content-Disposition", "").lower()
    return "text/html" in content_type and "attachment" not in content_disposition


def _follow_drive_confirmation(
    session: requests.Session,
    initial_response: requests.Response,
    file_id: str,
    timeout: int,
) -> requests.Response:
    if not _response_is_html(initial_response):
        return initial_response

    page_html = initial_response.text
    soup = BeautifulSoup(page_html, "html.parser")
    form = soup.find("form", {"id": "download-form"})
    if form and form.get("action"):
        action_url = str(form.get("action"))
        form_params = {}
        for field in form.find_all("input"):
            name = field.get("name")
            if name:
                form_params[name] = field.get("value", "")
        form_params.setdefault("id", file_id)
        form_params.setdefault("export", "download")

        initial_response.close()
        confirmed = session.get(action_url, params=form_params, stream=True, timeout=timeout)
        confirmed.raise_for_status()
        return confirmed

    for cookie_key, cookie_value in initial_response.cookies.items():
        if cookie_key.startswith("download_warning"):
            initial_response.close()
            confirmed = session.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": file_id, "confirm": cookie_value},
                stream=True,
                timeout=timeout,
            )
            confirmed.raise_for_status()
            return confirmed

    return initial_response


def _stream_to_file(response: requests.Response, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    try:
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

        # A valid WELFake CSV is much larger than this.
        if temp_path.stat().st_size < 1024:
            raise RuntimeError("Downloaded dataset looks incomplete (file is too small).")

        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def download_welfake_dataset(
    destination_path: Optional[Path] = None,
    source_url: Optional[str] = None,
    timeout: int = 120,
) -> Path:
    target = Path(destination_path) if destination_path else DEFAULT_DATASET_PATH
    drive_url = source_url or get_welfake_dataset_source_url()
    file_id = _extract_google_drive_file_id(drive_url)

    try:
        with requests.Session() as session:
            initial = session.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": file_id},
                stream=True,
                timeout=timeout,
            )
            initial.raise_for_status()

            resolved = _follow_drive_confirmation(session, initial, file_id=file_id, timeout=timeout)
            if _response_is_html(resolved):
                raise RuntimeError(
                    "Google Drive returned an HTML page instead of the dataset file. "
                    "Ensure the file is shared as 'Anyone with the link: Viewer'."
                )

            _stream_to_file(resolved, target)
            resolved.close()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download WELFake dataset from Google Drive: {exc}") from exc

    return target


def ensure_welfake_dataset(dataset_path: Optional[Path] = None, force_download: bool = False) -> Path:
    if dataset_path:
        target = Path(dataset_path)
        if target.exists() and not force_download:
            return target
        return download_welfake_dataset(destination_path=target)

    if DEFAULT_DATASET_PATH.exists() and not force_download:
        return DEFAULT_DATASET_PATH

    if LEGACY_DATASET_PATH.exists() and not force_download:
        return LEGACY_DATASET_PATH

    return download_welfake_dataset(destination_path=DEFAULT_DATASET_PATH)


def get_welfake_dataset_status() -> Dict[str, object]:
    if DEFAULT_DATASET_PATH.exists():
        current_path = DEFAULT_DATASET_PATH
    elif LEGACY_DATASET_PATH.exists():
        current_path = LEGACY_DATASET_PATH
    else:
        current_path = DEFAULT_DATASET_PATH

    return {
        "available_locally": current_path.exists(),
        "local_path": str(current_path),
        "cache_path": str(DEFAULT_DATASET_PATH),
        "legacy_path": str(LEGACY_DATASET_PATH),
        "source_url": get_welfake_dataset_source_url(),
    }


def load_welfake_data(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"WELFake dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, usecols=["title", "text", "label"])
    df = df.dropna(subset=["title", "text", "label"])
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df


def _normalize_for_training(text: str) -> str:
    lowered = str(text).lower()
    lowered = re.sub(r"http\S+|www\S+", " ", lowered)
    lowered = re.sub(r"[^a-z\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def prepare_text_and_labels(df: pd.DataFrame):
    combined = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    combined = combined.str.slice(stop=6000)
    cleaned = combined.apply(_normalize_for_training)
    mask = cleaned.str.len() > 0
    X = cleaned[mask]
    y = df.loc[mask, "label"]
    return X, y


def train_welfake_model(
    profile: str = "quick",
    dataset_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    vectorizer_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: Optional[int] = None,
) -> Dict[str, object]:
    profile_name = _validate_profile(profile)
    preexisting_paths = {
        path
        for path in (Path(dataset_path) if dataset_path else None, DEFAULT_DATASET_PATH, LEGACY_DATASET_PATH)
        if path is not None and path.exists()
    }
    dataset = ensure_welfake_dataset(dataset_path=Path(dataset_path) if dataset_path else None)
    dataset_downloaded = dataset not in preexisting_paths
    model_output = Path(model_path) if model_path else PROFILE_MODEL_PATHS[profile_name]
    vectorizer_output = Path(vectorizer_path) if vectorizer_path else PROFILE_VECTORIZER_PATHS[profile_name]
    metrics_output = Path(metrics_path) if metrics_path else PROFILE_METRICS_PATHS[profile_name]

    resolved_max_rows = PROFILE_MAX_ROWS[profile_name] if max_rows is None else max_rows

    df = load_welfake_data(dataset)
    raw_rows_available = int(len(df))

    if resolved_max_rows and len(df) > resolved_max_rows:
        df, _ = train_test_split(
            df,
            train_size=resolved_max_rows,
            random_state=random_state,
            stratify=df["label"],
        )

    X, y = prepare_text_and_labels(df)

    if len(X) < 1000:
        raise ValueError("Dataset is too small after preprocessing. Need at least 1000 non-empty rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=280, solver="saga", random_state=random_state)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    probabilities = model.predict_proba(X_test_vec)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": [float(x) for x in fpr.tolist()],
        "tpr": [float(x) for x in tpr.tolist()],
        "label_mapping": {"0": "REAL", "1": "FAKE"},
        "profile": profile_name,
        "dataset_path": str(dataset),
        "dataset_downloaded_for_run": dataset_downloaded,
        "raw_rows_available": raw_rows_available,
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "max_rows_setting": int(resolved_max_rows) if resolved_max_rows else None,
    }

    model_output.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_output)
    joblib.dump(vectorizer, vectorizer_output)
    with metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with ACTIVE_PROFILE_PATH.open("w", encoding="utf-8") as handle:
        handle.write(profile_name)

    with ACTIVE_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "profile": profile_name,
        "model_path": str(model_output),
        "vectorizer_path": str(vectorizer_output),
        "metrics_path": str(metrics_output),
        "active_metrics_path": str(ACTIVE_METRICS_PATH),
        "active_profile_path": str(ACTIVE_PROFILE_PATH),
        "dataset_path": str(dataset),
        "dataset_downloaded": dataset_downloaded,
        "metrics": metrics,
    }


def main() -> None:
    result = train_welfake_model()
    metrics = result["metrics"]
    print(f"Trained model on WELFake dataset ({result['profile']} profile)")
    print(f"Model: {result['model_path']}")
    print(f"Vectorizer: {result['vectorizer_path']}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"F1: {metrics['f1'] * 100:.2f}%")


if __name__ == "__main__":
    main()
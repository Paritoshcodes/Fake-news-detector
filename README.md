# Fake News Radar

A polished Streamlit app for fake news classification with a single-page tabbed UX, URL content extraction, and profile-based model training on the WELFake dataset.

## Highlights

- Single-page tab navigation (`Home`, `Text Analyzer`, `URL Analyzer`, `Model Insights`) in `app.py`
- Text classification with confidence scoring
- URL analysis using both article title and main body content
- WELFake training profiles:
   - `quick`: stratified subset for faster training
   - `full`: full dataset training
- Automatic dataset download from Google Drive when training starts (no large CSV committed to GitHub)
- Profile-specific saved artifacts with active profile switching
- Model metrics dashboard (Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix)

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas / numpy
- matplotlib
- BeautifulSoup + requests

## Repository Layout

```
fake-news-streamlit-app/
|-- app.py
|-- artifacts/
|   |-- datasets/
|   |   `-- WELFake_Dataset.csv   # downloaded at runtime on first training run
|   |-- fake_news_model_quick.joblib
|   |-- fake_news_model_full.joblib
|   |-- tfidf_vectorizer_quick.joblib
|   |-- tfidf_vectorizer_full.joblib
|   |-- training_metrics_quick.json
|   |-- training_metrics_full.json
|   |-- training_metrics_active.json
|   `-- active_profile.txt
|-- src/
|   |-- models/
|   |   |-- predict.py
|   |   `-- train.py
|   |-- ui/
|   |   |-- components.py
|   |   `-- theme.py
|   `-- utils/
|       `-- web_scraper.py
|-- requirements.txt
`-- README.md
```

## Setup

1. Clone the repo

```bash
git clone <your-repo-url>
cd fake-news-streamlit-app
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

## How Training Works

Use the `Model Insights` tab and click `Train on WELFake` with either `quick` or `full` selected.

If the dataset is not already cached locally, the app downloads it automatically from Google Drive:

- `https://drive.google.com/file/d/13lcNYSvVfJhC5xl-84k5AcHvNVnKiI1T/view?usp=drive_link`

The downloaded file is cached at:

- `artifacts/datasets/WELFake_Dataset.csv`

You can override the dataset URL in deployment environments using:

- `WELFAKE_DATASET_URL`

- Quick profile writes:
   - `artifacts/fake_news_model_quick.joblib`
   - `artifacts/tfidf_vectorizer_quick.joblib`
   - `artifacts/training_metrics_quick.json`
- Full profile writes:
   - `artifacts/fake_news_model_full.joblib`
   - `artifacts/tfidf_vectorizer_full.joblib`
   - `artifacts/training_metrics_full.json`

The active profile is tracked in:

- `artifacts/active_profile.txt`
- `artifacts/training_metrics_active.json`

Inference always uses the active profile artifacts.

## Label Semantics

The WELFake label mapping used in training/inference is:

- `0 = REAL`
- `1 = FAKE`

## Notes for GitHub Push

- This repo is configured to ignore the `artifacts/` folder by default so trained model files are not uploaded unintentionally.
- `WELFake_Dataset.csv` does not need to be committed. It is downloaded at runtime from Google Drive during training.
- Ensure the Google Drive file is shared as **Anyone with the link (Viewer)** for Streamlit Cloud deployments.

## License

MIT
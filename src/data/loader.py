import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    return df

def load_and_preprocess_data(file_path):
    df = load_data(file_path)
    df = df[['title', 'text', 'label']].dropna()
    df['label'] = df['label'].map({0: 'FAKE', 1: 'REAL'})
    return df
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_MAPPING = {
    'Graduate': 0,
    'Dropout': 1,
    'Enrolled': 2
}

CAT_COLS = [
    "Course", "Father's occupation", "Father's qualification",
    "Mother's qualification", "Mother's occupation", "Application mode",
    "Marital status", "Nacionality", "Previous qualification"
]

RARE_CATEGORY_THRESHOLD = 50

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=';')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        print(f"Warning: dropped {dropped} rows with missing values")
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Target' in df.columns:
        df['Target'] = df['Target'].map(TARGET_MAPPING)
        unmapped = df['Target'].isna().sum()
        if unmapped > 0:
            raise ValueError(f"{unmapped} rows have an unrecognized Target value")
    return df

def remove_zero_grade_graduates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove graduated students who have 0 approved units on
    average across both semesters. (anomaly removal)"""
    mask = (
        df[['Curricular units 1st sem (approved)',
            'Curricular units 2nd sem (approved)']].mean(axis=1) == 0
    ) & (df['Target'] == 0)
    return df.drop(df[mask].index)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group categories that appear fewer than `threshold` times under a single
    label (99) to reduce dimensionality and avoid overfitting."""

    def __init__(self, cat_cols=CAT_COLS, threshold=RARE_CATEGORY_THRESHOLD):
        self.cat_cols = cat_cols
        self.threshold = threshold

    def fit(self, X, y=None):
        self.rare_categories_ = {}
        for col in self.cat_cols:
            counts = X[col].astype(str).value_counts()
            self.rare_categories_[col] = counts[counts < self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols:
            col_as_str = X[col].astype(str)
            X[col] = col_as_str.where(~col_as_str.isin(self.rare_categories_[col]), "99")
        return X


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=['Target'])
    y = df['Target']
    return X, y


def build_preprocessing_pipeline() -> Pipeline:
    """Builds a pipeline to preprocess the data."""
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CAT_COLS)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('rare_grouping', RareCategoryGrouper()),
        ('encoding', column_transformer),
    ])
    return pipeline


def load_and_split(file_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load, clean, filter anomalies and split the df"""
    df = load_data(file_path)
    df = clean_data(df)
    df = encode_target(df)
    df = remove_zero_grade_graduates(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "data.csv"

    try:
        X_train, X_test, y_train, y_test = load_and_split(data_path)

        pipeline = build_preprocessing_pipeline()
        pipeline.fit(X_train)

        X_train_enc = pipeline.transform(X_train)
        X_test_enc = pipeline.transform(X_test)

        print(f"Preprocessing OK — train: {X_train_enc.shape}, test: {X_test_enc.shape}")
    except Exception as e:
        print(f"Error during preprocessing execution: {e}")
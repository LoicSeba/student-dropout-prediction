import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

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
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path, sep=';')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values (There are no missing values but we keep this in case of a file change).
    """
    df = df.dropna()
    
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target variable 'Target' into numeric values:
    Graduate -> 0, Dropout -> 1, Enrolled -> 2
    """
    if 'Target' in df.columns:
        df['Target'] = df['Target'].map(TARGET_MAPPING)

    return df


def remove_zero_grade_graduates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove graduated students who have 0 approved units on
    average across both semesters.
    """
    mask = (
        df[['Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)']].mean(axis=1) == 0
    ) & (df['Target'] == 0)

    df = df.drop(df[mask].index)

    return df


def group_rare_categories(df: pd.DataFrame, cat_cols: list[str] = CAT_COLS, threshold: int = RARE_CATEGORY_THRESHOLD) -> pd.DataFrame:
    """
    Group categories that appear fewer than `threshold` times under a single
    label (99) to reduce dimensionality and avoid overfitting.
    """
    df = df.copy()
    for col in cat_cols:
        counts = df[col].value_counts()
        rares = counts[counts < threshold].index
        df[col] = df[col].replace(rares, 99)
    return df


def one_hot_encode(df: pd.DataFrame, cat_cols: list[str] = CAT_COLS):
    """
    One-hot encode categorical columns and split the dataframe into
    features (X) and target (y).
    """
    X = pd.get_dummies(df.drop(columns=['Target']), columns=cat_cols, drop_first=True)
    y = df['Target']
    return X, y


def preprocess_pipeline(file_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Run the full preprocessing pipeline (matching the notebook) and return
    X_train, X_test, y_train, y_test.
    """
    df = load_data(file_path)
    df = clean_data(df)
    df = encode_target(df)
    df = remove_zero_grade_graduates(df)
    df = group_rare_categories(df)

    X, y = one_hot_encode(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the preprocessing pipeline locally
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "data.csv"

    try:
        X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)
        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing execution: {e}")
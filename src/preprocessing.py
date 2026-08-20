import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path, sep=';')
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by renaming columns and handling missing values.
    """
    # Rename columns for better readability
    df = df.rename(columns={
        'Nacionality': 'Nationality',
        'Age at enrollment': 'Age'
    })
    
    # There is no missing values but we keep this in case of a new file
    df = df.dropna()
    
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target variable 'Target' into numeric values:
    Dropout -> 0, Enrolled -> 1, Graduate -> 2
    """
    target_mapping = {
        'Dropout': 0,
        'Enrolled': 1,
        'Graduate': 2
    }
    
    if 'Target' in df.columns:
        df['Target'] = df['Target'].map(target_mapping)
    
    return df

def preprocess_pipeline(file_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Run the full preprocessing pipeline and return X_train, X_test, y_train, y_test.
    """
    df = load_data(file_path)
    df = clean_data(df)
    df = encode_target(df)
    
    X = df.drop(columns=['Target'])
    y = df['Target']
    
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
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
    except Exception as e:
        print(f"Error during preprocessing execution: {e}")
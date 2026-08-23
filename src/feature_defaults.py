import json
from pathlib import Path

import pandas as pd


def compute_feature_defaults(X_train: pd.DataFrame) -> dict:
    defaults = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            value = X_train[col].median()
            defaults[col] = round(float(value), 2)
        else:
            value = X_train[col].mode(dropna=True)
            defaults[col] = str(value.iloc[0]) if not value.empty else None
    return defaults


def save_feature_defaults(X_train: pd.DataFrame, output_path: Path) -> None:
    defaults = compute_feature_defaults(X_train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(defaults, f, indent=2)
    print(f"Saved feature defaults to: {output_path}")

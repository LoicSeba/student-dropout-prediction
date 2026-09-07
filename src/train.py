from pathlib import Path

import joblib
from joblib import Memory
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from evaluate import evaluate
from feature_defaults import save_feature_defaults
from preprocessing import build_preprocessing_pipeline, load_and_split

MODEL_CANDIDATES = {
    "random_forest": {
        "estimator": RandomForestClassifier(class_weight="balanced", random_state=42),
        "param_grid": {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 8, 12, 16, 20, 30],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5, 0.75],
            "model__criterion": ["gini", "entropy"],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__subsample": [0.7, 0.85, 1.0],
        }
    },
}

def build_full_pipeline(preprocessing_pipeline: Pipeline, estimator, cache_dir: str = None) -> Pipeline:
    """Builds a full pipeline to preprocess the data and train a model"""
    memory = Memory(location=cache_dir, verbose=0) if cache_dir else None
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing_pipeline),
            ("model", estimator),
        ],
        memory=memory,
    )

def run_grid_search(pipeline, param_grid, X_train, y_train, cv=5, sample_weight=None):
    """Runs a GridSearchCV to find the best hyperparameters"""

    fit_params = {}
    if sample_weight is not None:
        fit_params["model__sample_weight"] = sample_weight

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=60,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=False,
    )

    search.fit(X_train, y_train, **fit_params)

    best_params = search.best_params_

    fine_param_grid = {}
    for param, val in best_params.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            fine_param_grid[param] = [val * 0.8, val, val * 1.2] 
        else:
            fine_param_grid[param] = [val]

    fine_search = GridSearchCV(
        pipeline,
        param_grid=fine_param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    fine_search.fit(X_train, y_train, **fit_params)

    return fine_search


def train_all_candidates(X_train, y_train, X_test, y_test, cache_dir: str = None):
    """Trains all models and returns the results"""

    results = {}

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    for name, cfg in MODEL_CANDIDATES.items():
        print(f"\n=== Training {name} (GridSearchCV) ===")
        preprocessing_pipeline = build_preprocessing_pipeline()
        full_pipeline = build_full_pipeline(preprocessing_pipeline, clone(cfg["estimator"]), cache_dir=cache_dir)

        sw = weights if name == "gradient_boosting" else None
        grid = run_grid_search(full_pipeline, cfg["param_grid"], X_train, y_train, sample_weight=sw)
        print(f"Best CV macro F1: {grid.best_score_:.4f}")
        print(f"Best params: {grid.best_params_}")


        test_f1 = evaluate(grid.best_estimator_, X_test, y_test, label=name)

        results[name] = {
            "cv_f1": grid.best_score_,
            "test_f1": test_f1,
            "fitted_pipeline": grid.best_estimator_,
            "best_params": grid.best_params_,
        }

    return results


def select_best_model(results: dict) -> tuple[str, dict]:
    """Selects the best model based on CV macro F1."""
    best_name = max(results, key=lambda name: results[name]["cv_f1"])
    return best_name, results[best_name]


def main():
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "data.csv"
    model_output_path = current_dir.parent / "models" / "final_pipeline.joblib"
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
 
    cache_dir = current_dir.parent / ".pipeline_cache"
 
    X_train, X_test, y_train, y_test = load_and_split(data_path)
 
    results = train_all_candidates(X_train, y_train, X_test, y_test, cache_dir=str(cache_dir))
 
    best_name, best_result = select_best_model(results)
 
    print(f"\n=== Selected model: {best_name} ===")
    print(f"CV macro F1: {best_result['cv_f1']:.4f}")
    print(f"Test macro F1: {best_result['test_f1']:.4f}")
 
    joblib.dump(best_result["fitted_pipeline"], model_output_path)
    print(f"\nSaved full pipeline (preprocessing + model) to: {model_output_path}")
 
    defaults_output_path = current_dir.parent / "models" / "feature_defaults.json"
    save_feature_defaults(X_train, defaults_output_path)

if __name__ == "__main__":
    main()
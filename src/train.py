from pathlib import Path

import joblib
from joblib import Memory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from evaluate import evaluate
from feature_defaults import save_feature_defaults
from feature_selection import (
    BalancedGradientBoostingClassifier,
    PreprocessingFeatureSelector,
)
from preprocessing import build_preprocessing_pipeline, load_and_split

MODEL_CANDIDATES = {
    "random_forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
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
        "estimator": BalancedGradientBoostingClassifier(
            random_state=42
        ),
        "param_grid": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__subsample": [0.7, 0.85, 1.0],
        },
    }
}


def build_full_pipeline(preprocessing_pipeline: Pipeline, estimator, cache_dir: str = None) -> Pipeline:
    """Builds a full pipeline to preprocess the data and train a model"""
    selector = PreprocessingFeatureSelector(
        preprocessing_pipeline=preprocessing_pipeline,
        estimator=estimator,
        min_features=5,
        max_features=20,
        step=1,
        correlation_threshold=0.95,
        cv=3,
        random_state=42
    )

    memory = Memory(location=str(cache_dir), verbose=0) if cache_dir else None

    return Pipeline(
        steps=[
            ("feature_selection", selector),
            ("model", estimator),
        ],
        memory=memory,
    )

def build_local_grid(best_params, model_name):
    grid = {}

    if model_name == "random_forest":
        max_depth = best_params["model__max_depth"]
        if max_depth is None:
            grid["model__max_depth"] = [8, 12, None]
        else:
            grid["model__max_depth"] = sorted(set([
                max(4, max_depth - 2),
                max_depth,
                max_depth + 2
            ]))

        min_samples_split = best_params["model__min_samples_split"]
        grid["model__min_samples_split"] = sorted(set([
            max(2, min_samples_split - 2),
            min_samples_split,
            min_samples_split + 2
        ]))

        min_samples_leaf = best_params["model__min_samples_leaf"]
        grid["model__min_samples_leaf"] = sorted(set([
            max(1, min_samples_leaf - 1),
            min_samples_leaf,
            min_samples_leaf + 1
        ]))

        max_features = best_params["model__max_features"]
        if isinstance(max_features, float):
            grid["model__max_features"] = sorted(set([
                max(0.25, round(max_features - 0.1, 2)),
                round(max_features, 2),
                min(1.0, round(max_features + 0.1, 2))
            ]))
        else:
            if max_features == "sqrt":
                grid["model__max_features"] = ["sqrt", "log2"]
            elif max_features == "log2":
                grid["model__max_features"] = ["log2", "sqrt"]
            else:
                grid["model__max_features"] = [max_features, "sqrt"]

    elif model_name == "gradient_boosting":
        max_depth = best_params["model__max_depth"]
        grid["model__max_depth"] = sorted(set([
            max(1, max_depth - 1),
            max_depth,
            max_depth + 1
        ]))

        min_samples_split = best_params["model__min_samples_split"]
        grid["model__min_samples_split"] = sorted(set([
            max(2, min_samples_split - 2),
            min_samples_split,
            min_samples_split + 2
        ]))

        min_samples_leaf = best_params["model__min_samples_leaf"]
        grid["model__min_samples_leaf"] = sorted(set([
            max(1, min_samples_leaf - 1),
            min_samples_leaf,
            min_samples_leaf + 1
        ]))

        learning_rate = best_params["model__learning_rate"]
        grid["model__learning_rate"] = sorted(set([
            round(max(0.005, learning_rate * 0.75), 4),
            round(learning_rate, 4),
            round(min(1.0, learning_rate * 1.25), 4)
        ]))

    return grid

def run_grid_search(pipeline, param_grid, X_train, y_train, model_name):
    """Runs a RandomizedSearchCV followed by a local GridSearchCV."""

    n_iter = min(30, len(list(ParameterGrid(param_grid))))

    randomized_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=5,
        random_state=42,
        n_jobs=-1,
        refit=False
    )

    print(f"\n[1/2] Running RandomizedSearchCV for {model_name}...")
    randomized_search.fit(X_train, y_train)

    best_params = randomized_search.best_params_

    print("\nBest parameters from RandomizedSearchCV:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print(f"RandomizedSearchCV best Macro-F1: {randomized_search.best_score_:.4f}")

    local_grid = build_local_grid(
        best_params=best_params,
        model_name=model_name
    )

    n_grid_points = len(list(ParameterGrid(local_grid)))
    print(f"\n[2/2] Running GridSearchCV ({n_grid_points} combinations)...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=local_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters from GridSearchCV:")
    for key, value in grid_search.best_params_.items():
        print(f"  {key}: {value}")

    print(f"GridSearchCV best Macro-F1: {grid_search.best_score_:.4f}")

    return grid_search


def train_all_candidates(X_train, y_train, X_test, y_test, cache_dir: str = None):
    """Trains all models and returns the results"""

    results = {}

    for name, cfg in MODEL_CANDIDATES.items():
        print(f"\n=== Training {name} (RandomizedSearchCV + GridSearchCV) ===")

        preprocessing_pipeline = build_preprocessing_pipeline()

        full_pipeline = build_full_pipeline(
            preprocessing_pipeline,
            cfg["estimator"],
            cache_dir=cache_dir
        )

        search = run_grid_search(
            pipeline=full_pipeline,
            param_grid=cfg["param_grid"],
            X_train=X_train,
            y_train=y_train,
            model_name=name
        )

        test_f1 = evaluate(
            search.best_estimator_,
            X_test,
            y_test,
            label=name
        )

        results[name] = {
            "cv_f1": search.best_score_,
            "test_f1": test_f1,
            "fitted_pipeline": search.best_estimator_,
            "best_params": search.best_params_,
        }

        print(f"\n--- Summary for {name} ---")
        print(f"Best CV Macro-F1: {search.best_score_:.4f}")
        print(f"Test Macro-F1: {test_f1:.4f}")
        print(f"Best parameters: {search.best_params_}")

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
 
    best_pipeline = best_result["fitted_pipeline"]
    best_pipeline.memory = None

    joblib.dump(best_pipeline, model_output_path)

    print(f"\nSaved full pipeline (preprocessing + feature selection + model) to: {model_output_path}")

    defaults_output_path = current_dir.parent / "models" / "feature_defaults.json"
    save_feature_defaults(X_train, defaults_output_path)

if __name__ == "__main__":
    main()
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


class ModelBasedFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects original variables using model-based feature importance."""

    def __init__(self, estimator=None, feature_names=None, feature_groups=None, min_features: int = 5,
                 max_features: int = 20, step: int = 1, correlation_threshold: float = 0.95,
                 cv: int = 3, random_state: int = 42) -> None:
        self.estimator = estimator
        self.feature_names = feature_names
        self.feature_groups = feature_groups
        self.min_features = min_features
        self.max_features = max_features
        self.step = step
        self.correlation_threshold = correlation_threshold
        self.cv = cv
        self.random_state = random_state

    def _get_estimator(self):
        if self.estimator is not None:
            return clone(self.estimator)
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1
        )

    def _fit_estimator(self, estimator, X, y):
        if isinstance(estimator, GradientBoostingClassifier):
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y
            )
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            estimator.fit(X, y)
        return estimator

    def _get_variable_groups(self):
        if self.feature_groups is not None:
            return self.feature_groups
        raise ValueError("feature_groups must be provided for feature selection")

    def _get_variable_importances(self, X, y):
        estimator = self._get_estimator()
        estimator = self._fit_estimator(estimator, X, y)

        importances = estimator.feature_importances_
        groups = self._get_variable_groups()

        variable_importances = {}
        for variable, indices in groups.items():
            variable_importances[variable] = float(np.sum(importances[indices]))

        return variable_importances

    def _select_least_correlated(self, X, variables, groups):
        if len(variables) <= self.min_features:
            return variables

        selected = list(variables)
        correlation_matrix = np.corrcoef(X, rowvar=False)

        while len(selected) > self.max_features:
            to_remove = None
            highest_correlation = 0.0

            for i, variable_a in enumerate(selected):
                for variable_b in selected[i + 1:]:
                    indices_a = groups[variable_a]
                    indices_b = groups[variable_b]

                    correlations = correlation_matrix[np.ix_(indices_a, indices_b)]
                    correlation = float(np.nanmax(np.abs(correlations)))

                    if correlation > highest_correlation:
                        highest_correlation = correlation
                        to_remove = variable_b

            if to_remove is None or highest_correlation < self.correlation_threshold:
                break

            selected.remove(to_remove)

        return selected

    def _evaluate_variables(self, X, y, groups, variables):
        indices = [index for variable in variables for index in groups[variable]]
        X_selected = X[:, indices]

        scores = []
        cv = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        for train_idx, valid_idx in cv.split(X_selected, y):
            estimator = self._get_estimator()
            estimator = self._fit_estimator(
                estimator,
                X_selected[train_idx],
                y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            )

            y_valid = y.iloc[valid_idx] if hasattr(y, "iloc") else y[valid_idx]
            predictions = estimator.predict(X_selected[valid_idx])
            scores.append(f1_score(y_valid, predictions, average="macro"))

        return float(np.mean(scores))

    def fit(self, X, y) -> "ModelBasedFeatureSelector":
        X = np.asarray(X)
        self.feature_names_in_ = np.asarray(self.feature_names, dtype=object)
        groups = self._get_variable_groups()

        variable_importances = self._get_variable_importances(X, y)
        ranked_variables = sorted(
            variable_importances,
            key=variable_importances.get,
            reverse=True
        )

        max_features = min(self.max_features, len(ranked_variables))
        min_features = min(self.min_features, max_features)

        best_variables = ranked_variables[:min_features]
        best_score = self._evaluate_variables(
            X, y, groups, best_variables
        )

        for number in range(min_features + self.step, max_features + 1, self.step):
            variables = ranked_variables[:number]
            score = self._evaluate_variables(X, y, groups, variables)

            if score > best_score:
                best_score = score
                best_variables = variables

        best_variables = self._select_least_correlated(
            X,
            best_variables,
            groups
        )

        self.selected_variables_ = best_variables
        self.selected_indices_ = [
            index
            for variable in best_variables
            for index in groups[variable]
        ]
        self.best_score_ = best_score

        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X)
        return X[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        feature_names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )
        return feature_names[self.selected_indices_]


class PreprocessingFeatureSelector(BaseEstimator, TransformerMixin):
    """Fits preprocessing, regroup variables and feature selection."""

    def __init__(self, preprocessing_pipeline, estimator=None, min_features: int = 5,
                 max_features: int = 20, step: int = 1, correlation_threshold: float = 0.95,
                 cv: int = 3, random_state: int = 42) -> None:
        self.preprocessing_pipeline = preprocessing_pipeline
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.step = step
        self.correlation_threshold = correlation_threshold
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y) -> "PreprocessingFeatureSelector":
        self.preprocessing_pipeline_ = clone(self.preprocessing_pipeline)
        X_encoded = self.preprocessing_pipeline_.fit_transform(X, y)

        feature_names = self.preprocessing_pipeline_.get_feature_names_out()
        feature_names = np.asarray(feature_names, dtype=object)

        column_transformer = self.preprocessing_pipeline_.named_steps["encoding"]
        feature_groups = {}
        output_indices = column_transformer.output_indices_
        cat_encoder = column_transformer.named_transformers_["cat"]
        start = output_indices["cat"].start

        for column, categories, drop_idx in zip(column_transformer.transformers_[0][2], cat_encoder.categories_, cat_encoder.drop_idx_):
            n_features = len(categories) - (1 if drop_idx is not None else 0)
            feature_groups[column] = list(range(start, start + n_features))
            start += n_features

        remainder_columns = [
            column for column in column_transformer.feature_names_in_
            if column not in column_transformer.transformers_[0][2]
        ]
        start = output_indices["remainder"].start
        for column in remainder_columns:
            feature_groups[column] = [start]
            start += 1

        self.selector_ = ModelBasedFeatureSelector(
            estimator=self.estimator,
            feature_names=feature_names,
            feature_groups=feature_groups,
            min_features=self.min_features,
            max_features=self.max_features,
            step=self.step,
            correlation_threshold=self.correlation_threshold,
            cv=self.cv,
            random_state=self.random_state
        )
        self.selector_.fit(X_encoded, y)

        return self

    def transform(self, X) -> np.ndarray:
        X_encoded = self.preprocessing_pipeline_.transform(X)
        return self.selector_.transform(X_encoded)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.selector_.get_feature_names_out()


class BalancedGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosting classifier with balanced sample weights."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3,
                 min_samples_split: int = 2, min_samples_leaf: int = 1, subsample: float = 1.0,
                 random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y) -> "BalancedGradientBoostingClassifier":
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state
        )

        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y
        )

        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_

        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.feature_importances_
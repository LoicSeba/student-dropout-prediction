from sklearn.metrics import classification_report, f1_score
def evaluate(model, X_test, y_test, label: str) -> float:
    """Evaluates a model on the test set."""
    preds = model.predict(X_test)
    print(f"\n--- {label} — test set ---")
    print(classification_report(y_test, preds))
    return f1_score(y_test, preds, average="macro")

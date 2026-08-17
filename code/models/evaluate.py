"""Model evaluation on held-out test set."""
import json
import joblib
import numpy as np
from sklearn.metrics import classification_report

from code.models.ordinal_metrics import ordinal_report
from code.utils.config import MODELS_DIR

CLASS_NAMES = ("Low", "Medium", "High", "Critical")


def evaluate(model_path, X_test, y_test, out_json=None):
    """Evaluate a saved model and optionally save results to JSON."""
    # Load model pipeline from disk
    pipe = joblib.load(model_path)
    
    # Compute class probability distributions and predictions
    proba = pipe.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    
    # Generate ordinal evaluation report (Accuracy, QWK, Within-One, F2)
    rep = ordinal_report(y_test, pred, proba, CLASS_NAMES)
    rep["per_class_report"] = classification_report(
        y_test, pred, target_names=list(CLASS_NAMES), output_dict=True
    )
    
    # Save evaluation dictionary to JSON if output path provided
    if out_json:
        with open(out_json, "w") as f:
            json.dump(rep, f, indent=2, default=float)
            
    return rep


def get_model_metadata():
    """Return metadata about available models."""
    # Check existence of serialized model checkpoint files
    models = {}
    for name in ["logistic_regression", "random_forest"]:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = {"path": str(path), "exists": True}
        else:
            models[name] = {"path": str(path), "exists": False}
    return models
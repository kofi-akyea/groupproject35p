"""Logistic Regression model training."""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from code.data_prep.encode_features import build_preprocessor
from code.utils.config import MODELS_DIR


def build_logreg_pipeline(C: float = 0.5, class_weight: str = "balanced") -> Pipeline:
    """Build a logistic regression pipeline with preprocessing."""
    # Build feature preprocessor
    pre = build_preprocessor()
    
    # Initialize multiclass Logistic Regression estimator
    clf = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=C, class_weight=class_weight, n_jobs=-1
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train(X_train, y_train) -> Pipeline:
    """Train and persist logistic regression model."""
    # Fit pipeline on training dataset
    pipe = build_logreg_pipeline()
    pipe.fit(X_train, y_train)
    
    # Serialize model to disk
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "logistic_regression.joblib")
    print(f"Model saved to {MODELS_DIR / 'logistic_regression.joblib'}")
    return pipe
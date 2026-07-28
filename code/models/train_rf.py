"""Random Forest model training."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from code.data_prep.encode_features import build_preprocessor
from code.utils.config import MODELS_DIR


def build_rf_pipeline(n_estimators: int = 500, class_weight: str = "balanced") -> Pipeline:
    """Build a Random Forest pipeline with preprocessing."""
    pre = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, class_weight=class_weight, random_state=0
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train(X_train, y_train) -> Pipeline:
    """Train and persist random forest model."""
    pipe = build_rf_pipeline()
    pipe.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "random_forest.joblib")
    print(f"Model saved to {MODELS_DIR / 'random_forest.joblib'}")
    return pipe
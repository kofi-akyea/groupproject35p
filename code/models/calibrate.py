"""Model calibration using sigmoid calibration for better probability estimates."""
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from code.data_prep.encode_features import build_preprocessor
from code.utils.config import MODELS_DIR


def calibrate(model: Pipeline, X_train, y_train, name: str):
    """
    Calibrate a trained model using sigmoid calibration with cross-validation.
    
    Calibration improves the reliability of predicted probabilities, which is
    important for decision support systems where probability estimates
    inform risk management actions.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cal = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=cv)
    cal.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, MODELS_DIR / f"{name}_calibrated.joblib")
    print(f"Calibrated model saved to {MODELS_DIR / f'{name}_calibrated.joblib'}")
    return cal

"""Inference wrapper - loads model and generates predictions with explanations."""
import joblib
import numpy as np
import pandas as pd
import uuid

from code.explainability.shap_engine import shap_for_pipeline, CLASS_NAMES
from code.explainability.narrative_builder import build_narrative, top_features_from_shap
from code.api.schemas import ProjectFeatures, PredictResponse
from code.utils.config import MODELS_DIR

DEFAULT_MODEL = "random_forest"


def _to_dataframe(features: ProjectFeatures) -> pd.DataFrame:
    """Convert ProjectFeatures dataclass to a single-row DataFrame."""
    # Construct DataFrame from ProjectFeatures schema attributes
    df = pd.DataFrame([{
        "Project_Type": features.Project_Type,
        "Complexity_Score": features.Complexity_Score,
        "Methodology_Used": features.Methodology_Used,
        "Project_Phase": features.Project_Phase,
        "Team_Experience_Level": features.Team_Experience_Level,
        "Project_Manager_Experience": features.Project_Manager_Experience,
        "Resource_Availability": features.Resource_Availability,
        "Team_Turnover_Rate": features.Team_Turnover_Rate,
        "Requirement_Stability": features.Requirement_Stability,
        "Risk_Management_Maturity": features.Risk_Management_Maturity,
        "Change_Control_Maturity": features.Change_Control_Maturity,
        "Communication_Frequency": features.Communication_Frequency,
        "Stakeholder_Engagement_Level": features.Stakeholder_Engagement_Level,
        "Schedule_Pressure": features.Schedule_Pressure,
        "Budget_Utilization_Rate": features.Budget_Utilization_Rate,
        "Historical_Risk_Incidents": features.Historical_Risk_Incidents,
        "Vendor_Reliability_Score": features.Vendor_Reliability_Score,
        "Tech_Environment_Stability": features.Tech_Environment_Stability,
    }])
    
    # Cast categorical feature columns to string dtype to align with preprocessor expectation
    categorical_cols = [
        "Project_Type", "Methodology_Used", "Project_Phase",
        "Team_Experience_Level", "Project_Manager_Experience",
        "Requirement_Stability", "Risk_Management_Maturity", 
        "Change_Control_Maturity", "Tech_Environment_Stability"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df


def predict(features: ProjectFeatures, model_name: str = DEFAULT_MODEL) -> PredictResponse:
    """Run prediction and explanation for a single project.
    
    Note: SHAP explanations are only generated for Random Forest and XGBoost models.
    For other models, only predictions and probabilities are returned.
    """
    # Load serialized model pipeline from disk
    pipe = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    X = _to_dataframe(features)
    
    # Extract preprocessing pipeline and classifier steps
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    
    # Transform raw feature row and compute class probabilities
    Xt = pre.transform(X)
    probs = clf.predict_proba(Xt)[0]
    probabilities_pct = {CLASS_NAMES[k]: round(float(probs[k]) * 100, 2) for k in range(len(CLASS_NAMES))}
    head_idx = int(np.argmax(probs))
    head = CLASS_NAMES[head_idx]
    
    # Identify tree-based models capable of TreeExplainer acceleration
    from sklearn.ensemble import RandomForestClassifier
    try:
        import xgboost
        XGBClassifier = xgboost.XGBClassifier
    except ImportError:
        XGBClassifier = None
    
    is_tree_model = isinstance(clf, RandomForestClassifier)
    if XGBClassifier is not None:
        is_tree_model = is_tree_model or isinstance(clf, XGBClassifier)
    
    shap_values = None
    top_features = []
    narrative = ""
    
    # Compute SHAP explanation values and generate natural language narrative if model is tree-based
    if is_tree_model:
        try:
            out = shap_for_pipeline(pipe, X)
            shap_values = out["head_shap"]
            top = top_features_from_shap(out["head_shap"], k=5)
            top_features = [{"feature": k, "shap": v} for k, v in top]
            narrative = build_narrative(head, top, probabilities_pct)
        except Exception as e:
            # Fall back to narrative message if SHAP computation fails
            narrative = f"SHAP explanation unavailable: {str(e)}"
    else:
        narrative = f"SHAP explanations are only available for Random Forest and XGBoost models. {model_name} predictions are provided without feature attribution."
    
    # Return structured PredictResponse dataclass object
    return PredictResponse(
        prediction=head,
        probabilities=probabilities_pct,
        shap=shap_values,
        top_features=top_features,
        narrative=narrative,
        request_id=str(uuid.uuid4()),
    )


def predict_batch(df: pd.DataFrame, model_name: str = DEFAULT_MODEL) -> list:
    """Run batch predictions without SHAP for faster processing.
    
    Validates input data before transformation to provide clear error messages.
    """
    from code.utils.config import FEATURE_NAMES
    
    # Check that input dataframe contains all required feature columns
    missing_cols = set(FEATURE_NAMES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Load model pipeline
    pipe = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    
    # Ensure categorical columns are strings
    categorical_cols = [
        "Project_Type", "Methodology_Used", "Project_Phase",
        "Team_Experience_Level", "Project_Manager_Experience",
        "Requirement_Stability", "Risk_Management_Maturity", 
        "Change_Control_Maturity", "Tech_Environment_Stability"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Transform batch input dataframe
    try:
        Xt = pre.transform(df)
    except Exception as e:
        raise ValueError(f"Data transformation failed: {str(e)}. Please check that all categorical values are valid and numeric fields are within expected ranges.")
    
    # Compute predictions and class probability distributions across all rows
    probs = clf.predict_proba(Xt)
    preds = clf.predict(Xt)
    
    # Map predictions to class labels
    from code.utils.risk_levels import RISK_LEVELS
    results = []
    for i in range(len(df)):
        pred_class = RISK_LEVELS[preds[i]]
        prob_dict = {RISK_LEVELS[k]: round(float(probs[i][k]) * 100, 2) for k in range(len(RISK_LEVELS))}
        results.append({
            "prediction": pred_class,
            "probabilities": prob_dict
        })
    
    return results
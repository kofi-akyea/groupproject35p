"""Inference wrapper - loads model and generates predictions with explanations."""
import joblib
import pandas as pd
import uuid

from code.explainability.shap_engine import shap_for_pipeline
from code.explainability.narrative_builder import build_narrative, top_features_from_shap
from code.api.schemas import ProjectFeatures, PredictResponse
from code.utils.config import MODELS_DIR

DEFAULT_MODEL = "random_forest"


def _to_dataframe(features: ProjectFeatures) -> pd.DataFrame:
    """Convert ProjectFeatures dataclass to a single-row DataFrame."""
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
    
    # Ensure categorical columns are strings (not category dtype) to match training data
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
    """Run prediction and explanation for a single project."""
    pipe = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    X = _to_dataframe(features)
    out = shap_for_pipeline(pipe, X)
    head = out["head"]
    top = top_features_from_shap(out["head_shap"], k=5)
    narr = build_narrative(head, top, out["probabilities"])
    return PredictResponse(
        prediction=head,
        probabilities=out["probabilities"],
        shap=out["head_shap"],
        top_features=[{"feature": k, "shap": v} for k, v in top],
        narrative=narr,
        request_id=str(uuid.uuid4()),
    )
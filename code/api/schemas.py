"""Request/response schemas for the API."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProjectFeatures:
    """Input features for a project risk prediction."""
    # Project typology and complexity attributes
    Project_Type: str = "IT"
    Complexity_Score: float = 5.0
    Methodology_Used: str = "Agile"
    Project_Phase: str = "Execution"
    
    # Team and managerial experience attributes
    Team_Experience_Level: str = "Mixed"
    Project_Manager_Experience: str = "Mid-level PM"
    Resource_Availability: float = 0.7
    Team_Turnover_Rate: float = 0.15
    
    # Process maturity and communication attributes
    Requirement_Stability: str = "Moderate"
    Risk_Management_Maturity: str = "Formal"
    Change_Control_Maturity: str = "Formal"
    Communication_Frequency: float = 3.0
    
    # Stakeholder, budget, and operational risk metrics
    Stakeholder_Engagement_Level: float = 0.6
    Schedule_Pressure: float = 0.4
    Budget_Utilization_Rate: float = 0.9
    Historical_Risk_Incidents: int = 2
    Vendor_Reliability_Score: float = 0.8
    Tech_Environment_Stability: str = "Mixed"


@dataclass
class PredictResponse:
    """Response from a prediction request."""
    # Predicted risk level class name (Low, Medium, High, Critical)
    prediction: str
    # Class probability dictionary percentages
    probabilities: dict
    # Dictionary of feature SHAP values
    shap: dict
    # Ordered list of top SHAP feature contributions
    top_features: list
    # Automated plain-language narrative text explanation
    narrative: str
    # Pipeline version identifier
    model_version: str = "1.0.0"
    # Unique request UUID identifier
    request_id: Optional[str] = None
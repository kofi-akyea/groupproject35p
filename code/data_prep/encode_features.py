"""Feature encoding utilities."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ORDINAL_COLS_WITH_ORDER = {
    "Project_Phase":                ["Initiation", "Planning", "Execution", "Monitoring", "Closure"],
    "Team_Experience_Level":          ["Junior", "Mixed", "Senior", "Expert"],
    "Project_Manager_Experience":     ["Junior PM", "Mid-level PM", "Senior PM", "Certified PM"],
    "Requirement_Stability":         ["Volatile", "Moderate", "Stable"],
    "Risk_Management_Maturity":      ["None", "Basic", "Formal", "Advanced"],
    "Change_Control_Maturity":       ["None", "Basic", "Formal", "Advanced"],
    "Tech_Environment_Stability":    ["Legacy/Unstable", "Mixed", "Modern/Stable"],
}

NOMINAL_COLS = ["Project_Type", "Methodology_Used"]

NUMERIC_COLS = [
    "Complexity_Score", "Stakeholder_Engagement_Level", "Resource_Availability",
    "Team_Turnover_Rate", "Budget_Utilization_Rate", "Communication_Frequency",
    "Schedule_Pressure", "Vendor_Reliability_Score", "Historical_Risk_Incidents",
]

TARGET_COL = "Risk_Level"
TARGET_ORDER = ["Low", "Medium", "High", "Critical"]


def build_preprocessor():
    """Returns a ColumnTransformer that one-hots nominals and ordinates ordinals."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ord_enc = OrdinalEncoder(categories=[ORDINAL_COLS_WITH_ORDER[c] for c in ORDINAL_COLS_WITH_ORDER])
    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("ord", ord_enc, list(ORDINAL_COLS_WITH_ORDER.keys())),
            ("nom", ohe, NOMINAL_COLS),
        ],
        remainder="drop",
    )
    return ct


def encode_target(y: pd.Series) -> pd.Series:
    """Encode the four-level ordinal target as integers 0..3."""
    mapping = {lvl: i for i, lvl in enumerate(TARGET_ORDER)}
    return y.map(mapping).astype(int)


def decode_target(y: pd.Series) -> pd.Series:
    """Decode integer predictions back to risk level strings."""
    mapping = {i: lvl for i, lvl in enumerate(TARGET_ORDER)}
    return y.map(mapping)
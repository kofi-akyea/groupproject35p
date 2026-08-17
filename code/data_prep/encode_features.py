"""Feature encoding utilities."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Prescribed ordinal category hierarchy order for ordinal encoding
ORDINAL_COLS_WITH_ORDER = {
    "Project_Phase":                ["Initiation", "Planning", "Execution", "Monitoring", "Closure"],
    "Team_Experience_Level":          ["Junior", "Mixed", "Senior", "Expert"],
    "Project_Manager_Experience":     ["Junior PM", "Mid-level PM", "Senior PM", "Certified PM"],
    "Requirement_Stability":         ["Volatile", "Moderate", "Stable"],
    "Risk_Management_Maturity":      ["None", "Basic", "Formal", "Advanced"],
    "Change_Control_Maturity":       ["None", "Basic", "Formal", "Advanced"],
    "Tech_Environment_Stability":    ["Legacy/Unstable", "Mixed", "Modern/Stable"],
}

# Nominal categorical columns for one-hot encoding
NOMINAL_COLS = ["Project_Type", "Methodology_Used"]

# Numeric features for median imputation and standard scaling
NUMERIC_COLS = [
    "Complexity_Score", "Stakeholder_Engagement_Level", "Resource_Availability",
    "Team_Turnover_Rate", "Budget_Utilization_Rate", "Communication_Frequency",
    "Schedule_Pressure", "Vendor_Reliability_Score", "Historical_Risk_Incidents",
]

# Ordinal risk target variable order
TARGET_COL = "Risk_Level"
TARGET_ORDER = ["Low", "Medium", "High", "Critical"]


def build_preprocessor():
    """Returns a ColumnTransformer that imputes, scales, one-hots nominals and encodes ordinals."""
    # Numeric pipeline: impute missing values with median, then standardize (z-score scaling)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Ordinal pipeline: impute missing values with mode, then encode using explicit category hierarchy
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[ORDINAL_COLS_WITH_ORDER[c] for c in ORDINAL_COLS_WITH_ORDER], handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    # Nominal pipeline: impute missing values with mode, then one-hot encode binary indicator columns
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine feature pipelines into a single ColumnTransformer object
    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("ord", ordinal_pipeline, list(ORDINAL_COLS_WITH_ORDER.keys())),
            ("nom", nominal_pipeline, NOMINAL_COLS),
        ],
        remainder="drop",
    )
    return ct


def encode_target(y: pd.Series) -> pd.Series:
    """Encode the four-level ordinal target as integers 0..3."""
    # Map string labels (Low, Medium, High, Critical) to integer indices [0, 1, 2, 3]
    mapping = {lvl: i for i, lvl in enumerate(TARGET_ORDER)}
    return y.map(mapping).astype(int)


def decode_target(y: pd.Series) -> pd.Series:
    """Decode integer predictions back to risk level strings."""
    # Map integer indices back to string labels
    mapping = {i: lvl for i, lvl in enumerate(TARGET_ORDER)}
    return y.map(mapping)
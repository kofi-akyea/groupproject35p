"""Data cleaning utilities."""
import pandas as pd
import numpy as np

# Define lists of numeric feature columns for range clipping and type coercion
NUMERIC_COLS = [
    "Complexity_Score", "Stakeholder_Engagement_Level", "Resource_Availability",
    "Team_Turnover_Rate", "Budget_Utilization_Rate", "Communication_Frequency",
    "Schedule_Pressure", "Vendor_Reliability_Score", "Historical_Risk_Incidents",
]

# Stakeholder_Engagement_Level ordinal string mapping to numeric score [0.0, 1.0]
SE_LEVEL_MAP = {"Low": 0.25, "Medium": 0.50, "High": 0.75, "Excellent": 1.0, "Poor": 0.0}

# Define lists of categorical feature columns for text sanitization
CATEGORICAL_COLS = [
    "Project_Type", "Methodology_Used", "Project_Phase",
    "Team_Experience_Level", "Project_Manager_Experience",
    "Requirement_Stability", "Risk_Management_Maturity", "Change_Control_Maturity",
    "Tech_Environment_Stability",
]


def clean(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Clean the raw dataframe: handle missing values, clip ranges, deduplicate.
    
    IMPORTANT: This function applies only deterministic transformations (range clipping,
    text standardization, deduplication). Data-dependent transformations like
    median/mode imputation and scaling are handled in the preprocessing pipeline
    to avoid data leakage from test/validation sets.
    
    Args:
        df: Input dataframe
        require_target: If True, drop rows missing Risk_Level. If False, keep all rows (for prediction-only data).
    """
    df = df.copy()

    # Drop rows with missing target label if training/evaluating
    if require_target and "Risk_Level" in df.columns:
        df = df.dropna(subset=["Risk_Level"]).reset_index(drop=True)

    # Convert Stakeholder_Engagement_Level string labels to numeric scale
    if "Stakeholder_Engagement_Level" in df.columns:
        df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].astype(str)
        df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].map(SE_LEVEL_MAP).fillna(0.5)
        df["Stakeholder_Engagement_Level"] = pd.to_numeric(df["Stakeholder_Engagement_Level"], errors="coerce")

    # Clip numeric features to valid physical domain boundaries
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c == "Complexity_Score":
                df[c] = df[c].clip(0.0, 10.0)
            elif c == "Stakeholder_Engagement_Level":
                df[c] = df[c].clip(0.0, 1.0)
            elif c == "Resource_Availability":
                df[c] = df[c].clip(0.0, 1.0)
            elif c == "Team_Turnover_Rate":
                df[c] = df[c].clip(0.0, 1.0)
            elif c == "Budget_Utilization_Rate":
                df[c] = df[c].clip(0.0, 1.5)
            elif c == "Communication_Frequency":
                df[c] = df[c].clip(0.0, 10.0)
            elif c == "Vendor_Reliability_Score":
                df[c] = df[c].clip(0.0, 1.0)
            elif c == "Schedule_Pressure":
                df[c] = df[c].clip(0.0, 1.0)
            elif c == "Historical_Risk_Incidents":
                df[c] = df[c].clip(0, 50).round().astype(int)

    # Standardize categorical text whitespace and set pandas category dtype
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object")
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].astype("category")

    # Drop duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    return df
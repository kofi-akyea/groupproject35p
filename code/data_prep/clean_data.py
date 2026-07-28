"""Data cleaning utilities."""
import pandas as pd
import numpy as np

NUMERIC_COLS = [
    "Complexity_Score", "Stakeholder_Engagement_Level", "Resource_Availability",
    "Team_Turnover_Rate", "Budget_Utilization_Rate", "Communication_Frequency",
    "Schedule_Pressure", "Vendor_Reliability_Score", "Historical_Risk_Incidents",
]

# Stakeholder_Engagement_Level needs special handling as it comes as categorical
SE_LEVEL_MAP = {"Low": 0.25, "Medium": 0.50, "High": 0.75, "Excellent": 1.0, "Poor": 0.0}

CATEGORICAL_COLS = [
    "Project_Type", "Methodology_Used", "Project_Phase",
    "Team_Experience_Level", "Project_Manager_Experience",
    "Requirement_Stability", "Risk_Management_Maturity", "Change_Control_Maturity",
    "Tech_Environment_Stability",
]


def clean(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Clean the raw dataframe: handle missing values, clip ranges, deduplicate.
    
    Args:
        df: Input dataframe
        require_target: If True, drop rows missing Risk_Level. If False, keep all rows (for prediction-only data).
    """
    df = df.copy()

    # Drop rows missing the target only if required
    if require_target and "Risk_Level" in df.columns:
        df = df.dropna(subset=["Risk_Level"]).reset_index(drop=True)

    # Convert Stakeholder_Engagement_Level from categorical to numeric FIRST
    # Dataset has values: "Low", "Medium", "High", "Excellent", "Poor"
    if "Stakeholder_Engagement_Level" in df.columns:
        # Convert to string type first (handles ArrowStringArray)
        df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].astype(str)
        df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].map(SE_LEVEL_MAP).fillna(0.5)
        # Ensure it's numeric type
        df["Stakeholder_Engagement_Level"] = pd.to_numeric(df["Stakeholder_Engagement_Level"], errors="coerce")

    # Impute numeric NaNs with median, clip to valid ranges
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
            # Clip to valid ranges
            if c == "Complexity_Score":
                df[c] = df[c].clip(0.0, 10.0)
            elif c in ("Stakeholder_Engagement_Level", "Resource_Availability",
                       "Team_Turnover_Rate", "Vendor_Reliability_Score",
                       "Schedule_Pressure"):
                df[c] = df[c].clip(0.0, 1.0)
            elif c == "Budget_Utilization_Rate":
                df[c] = df[c].clip(0.0, 1.5)
            elif c == "Communication_Frequency":
                df[c] = df[c].clip(0.0, 10.0)
            elif c == "Historical_Risk_Incidents":
                df[c] = df[c].clip(0, 50).round().astype(int)

    # Impute categorical NaNs with mode
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object")
            if df[c].isna().any():
                mode_val = df[c].mode()
                if len(mode_val) > 0:
                    df[c] = df[c].fillna(mode_val.iloc[0])
                else:
                    # Fallback to most common value from config if mode fails
                    from code.utils.config import FEATURE_SCHEMA
                    if c in FEATURE_SCHEMA:
                        df[c] = df[c].fillna(FEATURE_SCHEMA[c].default)
                    else:
                        df[c] = df[c].fillna("Unknown")
            df[c] = df[c].astype("category")

    # Final check: ensure no NaN values remain in feature columns
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + ["Stakeholder_Engagement_Level"]
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            # Force fill remaining NaNs
            if col in NUMERIC_COLS or col == "Stakeholder_Engagement_Level":
                df[col] = df[col].fillna(df[col].median() if df[col].dtype in ['float64', 'int64'] else 0.5)
            else:
                from code.utils.config import FEATURE_SCHEMA
                default_val = FEATURE_SCHEMA[col].default if col in FEATURE_SCHEMA else "Unknown"
                df[col] = df[col].fillna(default_val)

    # Deduplicate
    df = df.drop_duplicates().reset_index(drop=True)

    return df
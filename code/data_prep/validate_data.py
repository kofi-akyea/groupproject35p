"""Data validation schema utilities."""
import pandas as pd

# Valid category lists for feature schema validation
VALID_PROJECT_TYPES = ["Construction", "Manufacturing", "IT", "R&D", "Healthcare", "Marketing"]
VALID_METHODOLOGIES = ["Waterfall", "Kanban", "Agile", "Scrum", "Hybrid"]
VALID_PHASES = ["Planning", "Execution", "Initiation", "Closure", "Monitoring"]
VALID_TEAM_EXP = ["Senior", "Mixed", "Junior", "Expert"]
VALID_PM_EXP = ["Mid-level PM", "Certified PM", "Senior PM", "Junior PM"]
VALID_REQ_STABILITY = ["Moderate", "Stable", "Volatile"]
VALID_RISK_MGMT = ["None", "Basic", "Formal", "Advanced"]
VALID_CHANGE_CTRL = ["None", "Basic", "Formal", "Advanced"]
VALID_TECH_ENV = ["Legacy/Unstable", "Mixed", "Modern/Stable"]
VALID_RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

# Expected numeric range boundaries
NUMERIC_RANGES = {
    "Complexity_Score": (0.0, 10.0),
    "Stakeholder_Engagement_Level": (0.0, 1.0),
    "Resource_Availability": (0.0, 1.0),
    "Team_Turnover_Rate": (0.0, 1.0),
    "Budget_Utilization_Rate": (0.0, 1.5),
    "Communication_Frequency": (0.0, 10.0),
    "Schedule_Pressure": (0.0, 1.0),
    "Vendor_Reliability_Score": (0.0, 1.0),
    "Historical_Risk_Incidents": (0, 50),
}


def validate_data(df: pd.DataFrame) -> list:
    """Validate the dataset and return a list of error messages.
    Returns empty list if valid."""
    errors = []

    # Check required columns presence in input DataFrame
    required = ["Risk_Level", "Project_Type", "Complexity_Score", "Methodology_Used",
                "Project_Phase", "Team_Experience_Level", "Project_Manager_Experience",
                "Resource_Availability", "Team_Turnover_Rate", "Requirement_Stability",
                "Risk_Management_Maturity", "Change_Control_Maturity",
                "Communication_Frequency", "Schedule_Pressure", "Budget_Utilization_Rate",
                "Historical_Risk_Incidents", "Vendor_Reliability_Score",
                "Tech_Environment_Stability", "Stakeholder_Engagement_Level"]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return errors

    # Validate target Risk_Level values
    invalid_risk = df[~df["Risk_Level"].isin(VALID_RISK_LEVELS)]["Risk_Level"].unique()
    if len(invalid_risk) > 0:
        errors.append(f"Invalid Risk_Level values found: {invalid_risk.tolist()}")

    # Validate categorical column values against schema definitions
    cat_validations = [
        ("Project_Type", VALID_PROJECT_TYPES),
        ("Methodology_Used", VALID_METHODOLOGIES),
        ("Project_Phase", VALID_PHASES),
        ("Team_Experience_Level", VALID_TEAM_EXP),
        ("Project_Manager_Experience", VALID_PM_EXP),
        ("Requirement_Stability", VALID_REQ_STABILITY),
        ("Risk_Management_Maturity", VALID_RISK_MGMT),
        ("Change_Control_Maturity", VALID_CHANGE_CTRL),
        ("Tech_Environment_Stability", VALID_TECH_ENV),
    ]
    for col, valid_vals in cat_validations:
        if col in df.columns:
            invalid = df[~df[col].isin(valid_vals)][col].unique()
            if len(invalid) > 0:
                errors.append(f"Invalid values in {col}: {invalid.tolist()}")

    # Validate numeric feature range boundaries
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col in df.columns:
            col_numeric = pd.to_numeric(df[col], errors="coerce")
            below = (col_numeric < lo).sum()
            above = (col_numeric > hi).sum()
            if below > 0:
                errors.append(f"{col}: {below} values below minimum {lo}")
            if above > 0:
                errors.append(f"{col}: {above} values above maximum {hi}")

    return errors
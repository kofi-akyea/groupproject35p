"""Error handling utilities for the API."""
from flask import jsonify


def error_response(message: str, code: int):
    """Create a standardised error response."""
    return jsonify({"error": message, "code": code}), code


def validate_payload(payload: dict) -> str | None:
    """Validate a prediction request payload. Returns None if valid, error string otherwise."""
    required_fields = [
        "Project_Type", "Complexity_Score", "Methodology_Used", "Project_Phase",
        "Team_Experience_Level", "Project_Manager_Experience", "Resource_Availability",
        "Team_Turnover_Rate", "Requirement_Stability", "Risk_Management_Maturity",
        "Change_Control_Maturity", "Communication_Frequency",
        "Stakeholder_Engagement_Level", "Schedule_Pressure", "Budget_Utilization_Rate",
        "Historical_Risk_Incidents", "Vendor_Reliability_Score",
        "Tech_Environment_Stability",
    ]
    for field in required_fields:
        if field not in payload:
            return f"Missing required field: {field}"

    # Validate Complexity_Score range
    cs = payload.get("Complexity_Score", 0)
    try:
        cs = float(cs)
        if cs < 0.0 or cs > 10.0:
            return "Complexity_Score must be in [0, 10]"
    except (ValueError, TypeError):
        return "Complexity_Score must be a numeric value"

    return None
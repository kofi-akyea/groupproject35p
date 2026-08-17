"""Configuration module - single source of truth for project paths and feature specs."""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Define project root directory and key subdirectory paths
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

@dataclass(frozen=True)
class FeatureSpec:
    """Dataclass specifying metadata, types, boundaries, and UI grouping for each input feature."""
    name: str
    display: str
    kind: str                    # "numeric" | "ordinal" | "nominal"
    order: Optional[List[str]] = None
    minv: Optional[float] = None
    maxv: Optional[float] = None
    step: Optional[float] = None
    default: Optional[object] = None
    group: str = "Project"

# Canonical feature specifications (18 regularized features)
FEATURE_SPECS: List[FeatureSpec] = [
    # Project characteristics group
    FeatureSpec("Project_Type", "Project Type", "nominal",
                order=["Construction", "IT", "Healthcare", "Manufacturing", "R&D", "Marketing"],
                default="IT", group="Project"),
    FeatureSpec("Complexity_Score", "Complexity Score", "numeric",
                minv=0.0, maxv=10.0, step=0.1, default=5.0, group="Project"),
    FeatureSpec("Methodology_Used", "Methodology", "nominal",
                order=["Waterfall", "Agile", "Scrum", "Kanban", "Hybrid"],
                default="Agile", group="Project"),
    FeatureSpec("Project_Phase", "Project Phase", "ordinal",
                order=["Initiation", "Planning", "Execution", "Monitoring", "Closure"],
                default="Execution", group="Project"),

    # Team & PM experience group
    FeatureSpec("Team_Experience_Level", "Team Experience", "ordinal",
                order=["Junior", "Mixed", "Senior", "Expert"],
                default="Mixed", group="Team"),
    FeatureSpec("Project_Manager_Experience", "Project Manager Experience", "ordinal",
                order=["Junior PM", "Mid-level PM", "Senior PM", "Certified PM"],
                default="Mid-level PM", group="Team"),
    FeatureSpec("Resource_Availability", "Resource Availability", "numeric",
                minv=0.0, maxv=1.0, step=0.01, default=0.7, group="Team"),
    FeatureSpec("Team_Turnover_Rate", "Team Turnover Rate", "numeric",
                minv=0.0, maxv=1.0, step=0.01, default=0.15, group="Team"),

    # Governance & process maturity group
    FeatureSpec("Requirement_Stability", "Requirement Stability", "ordinal",
                order=["Volatile", "Moderate", "Stable"],
                default="Moderate", group="Governance"),
    FeatureSpec("Risk_Management_Maturity", "Risk Management Maturity", "ordinal",
                order=["None", "Basic", "Formal", "Advanced"],
                default="Formal", group="Governance"),
    FeatureSpec("Change_Control_Maturity", "Change Control Maturity", "ordinal",
                order=["None", "Basic", "Formal", "Advanced"],
                default="Formal", group="Governance"),
    FeatureSpec("Communication_Frequency", "Communication Frequency", "numeric",
                minv=0.0, maxv=10.0, step=0.1, default=3.0, group="Governance"),

    # Schedule, budget, and environmental factors group
    FeatureSpec("Schedule_Pressure", "Schedule Pressure", "numeric",
                minv=0.0, maxv=1.0, step=0.01, default=0.4, group="Schedule/Budget/External"),
    FeatureSpec("Budget_Utilization_Rate", "Budget Utilisation Rate", "numeric",
                minv=0.0, maxv=1.5, step=0.01, default=0.9, group="Schedule/Budget/External"),
    FeatureSpec("Historical_Risk_Incidents", "Historical Risk Incidents", "numeric",
                minv=0, maxv=20, step=1, default=2, group="Schedule/Budget/External"),
    FeatureSpec("Vendor_Reliability_Score", "Vendor Reliability", "numeric",
                minv=0.0, maxv=1.0, step=0.01, default=0.8, group="Schedule/Budget/External"),
    FeatureSpec("Tech_Environment_Stability", "Tech Environment Stability", "ordinal",
                order=["Legacy/Unstable", "Mixed", "Modern/Stable"],
                default="Mixed", group="Schedule/Budget/External"),
    FeatureSpec("Stakeholder_Engagement_Level", "Stakeholder Engagement", "numeric",
                minv=0.0, maxv=1.0, step=0.01, default=0.6, group="Governance"),
]

# List of feature column names
FEATURE_NAMES = [f.name for f in FEATURE_SPECS]

# Feature schema map
FEATURE_SCHEMA = {f.name: f for f in FEATURE_SPECS}
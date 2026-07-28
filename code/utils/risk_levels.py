"""Single source of truth for risk level definitions, colours, and actions."""

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
RISK_RANK = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
RISK_COLOURS = {
    "Low":      "#2ecc71",
    "Medium":   "#f1c40f",
    "High":     "#e67e22",
    "Critical": "#c0392b",
}
RISK_ACTIONS = {
    "Low":      "Standard monitoring, quarterly risk review.",
    "Medium":   "Develop a risk response plan, bi-weekly tracking, assign a risk owner.",
    "High":     "Escalate to the PMO, weekly status reporting, review budget and schedule reserves, consider mitigation plan update.",
    "Critical": "Immediate executive review, consider scope, schedule, or strategy change, daily monitoring, risk owner at director level.",
}
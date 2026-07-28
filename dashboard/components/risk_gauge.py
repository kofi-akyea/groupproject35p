"""Risk gauge component using Plotly."""
import plotly.graph_objects as go
import streamlit as st

ZONES = [
    (0.00, 0.25, "#2ecc71", "Low"),
    (0.25, 0.50, "#f1c40f", "Medium"),
    (0.50, 0.75, "#e67e22", "High"),
    (0.75, 1.00, "#c0392b", "Critical"),
]


def render(value: float, label: str):
    """Render a risk gauge chart."""
    steps = []
    for lo, hi, col, _ in ZONES:
        steps.append({"range": [lo, hi], "color": col})
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%"},
        title={"text": f"Predicted Risk: {label}"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#34495e"},
            "steps": steps,
            "threshold": {
                "line": {"color": "#2c3e50", "width": 4},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    st.plotly_chart(fig, use_container_width=True)
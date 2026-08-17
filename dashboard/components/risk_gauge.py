"""Risk gauge component using Plotly."""
import plotly.graph_objects as go
import streamlit as st

# Color zones mapping risk percentage ranges to status colors
ZONES = [
    (0, 25, "#2563EB", "Low"),
    (25, 50, "#D97706", "Medium"),
    (50, 75, "#EA580C", "High"),
    (75, 100, "#DC2626", "Critical"),
]


def render(value: float, label: str):
    """Render a risk gauge chart."""
    # Define gauge background color steps for each risk tier
    steps = [{"range": [lo, hi], "color": col} for lo, hi, col, _ in ZONES]
    
    # Create gauge indicator chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 28, "color": "#0F172A", "family": "Inter, sans-serif"}},
        title={"text": f"Predicted Risk: <b>{label}</b>", "font": {"size": 16, "color": "#0F172A", "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
            "bar": {"color": "#0F172A"},
            "steps": steps,
            "threshold": {
                "line": {"color": "#0F172A", "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    
    # Configure transparent backgrounds and tight padding
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    # Render interactive plot in container
    st.plotly_chart(fig, use_container_width=True)
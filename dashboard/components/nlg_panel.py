"""Natural language explanation panel component."""
import streamlit as st


def render(narrative: str, top_features: list, risk_colour: str):
    """Render the NLG explanation panel."""
    st.markdown("<h4 style='margin: 0 0 12px 0; font-size: 16px; font-weight: 600;'>Plain-Language Explanation</h4>", unsafe_allow_html=True)
    
    # Format key SHAP feature drivers as clean bullet points
    feature_items = "".join(
        f"<li style='margin-bottom: 4px;'><strong>{f['feature']}</strong> (SHAP contribution: <span style='font-family: monospace;'>{f['shap']:+.2f}</span>)</li>"
        for f in top_features
    )
    
    # Render clean neutral card container without colored edge accent lines
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 8px;">Key Risk Drivers</div>
            <ul style="margin: 0 0 16px 0; padding-left: 20px; color: #0F172A; font-size: 14px;">
                {feature_items}
            </ul>
            <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 6px;">Automated Risk Narrative</div>
            <p style="margin: 0; color: #334155; font-size: 14px; line-height: 1.5;">{narrative}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
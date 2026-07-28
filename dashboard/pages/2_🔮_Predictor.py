"""Predictor page - Interactive project risk prediction with explanations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from code.utils.risk_levels import RISK_COLOURS, RISK_ACTIONS, RISK_RANK
from code.api.schemas import ProjectFeatures
from dashboard.components import input_form, risk_gauge, probability_bars, shap_waterfall, nlg_panel

st.set_page_config(page_title="Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Project Risk Predictor")
st.caption("Enter project parameters; the system returns a four-level risk prediction and explainable rationale.")

values = input_form.render()

col_pred, col_nlg = st.columns([1, 1.4])

with st.form("predict_form", clear_on_submit=False):
    cols = st.columns(3)
    with cols[0]:
        submitted = st.form_submit_button("🚀 Predict Risk", use_container_width=True)
    with cols[1]:
        reset = st.form_submit_button("🔄 Reset", use_container_width=True)

if reset:
    st.rerun()

if submitted:
    # Ensure all values are properly typed
    payload = {
        "Project_Type": str(values["Project_Type"]),
        "Complexity_Score": float(values["Complexity_Score"]),
        "Methodology_Used": str(values["Methodology_Used"]),
        "Project_Phase": str(values["Project_Phase"]),
        "Team_Experience_Level": str(values["Team_Experience_Level"]),
        "Project_Manager_Experience": str(values["Project_Manager_Experience"]),
        "Resource_Availability": float(values["Resource_Availability"]),
        "Team_Turnover_Rate": float(values["Team_Turnover_Rate"]),
        "Requirement_Stability": str(values["Requirement_Stability"]),
        "Risk_Management_Maturity": str(values["Risk_Management_Maturity"]),
        "Change_Control_Maturity": str(values["Change_Control_Maturity"]),
        "Communication_Frequency": float(values["Communication_Frequency"]),
        "Stakeholder_Engagement_Level": float(values["Stakeholder_Engagement_Level"]),
        "Schedule_Pressure": float(values["Schedule_Pressure"]),
        "Budget_Utilization_Rate": float(values["Budget_Utilization_Rate"]),
        "Historical_Risk_Incidents": int(values["Historical_Risk_Incidents"]),
        "Vendor_Reliability_Score": float(values["Vendor_Reliability_Score"]),
        "Tech_Environment_Stability": str(values["Tech_Environment_Stability"]),
    }

    try:
        from code.api.inference import predict
        result = predict(ProjectFeatures(**payload), model_name="random_forest")
        st.session_state["last_result"] = result
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

res = st.session_state.get("last_result")
if res:
    head = res.prediction
    colour = RISK_COLOURS.get(head, "#888")
    with col_pred:
        prob_for_head = res.probabilities.get(head, 0.0)
        risk_gauge.render(prob_for_head, head)
        probability_bars.render(res.probabilities)
        action = RISK_ACTIONS.get(head, "Review project details.")
        st.markdown(
            f"<div style='background:{colour};color:white;padding:10px;border-radius:6px;text-align:center'>"
            f"<b>Recommended Action:</b> {action}</div>",
            unsafe_allow_html=True,
        )
    with col_nlg:
        nlg_panel.render(res.narrative, res.top_features, colour)
        shap_waterfall.render([(f["feature"], f["shap"]) for f in res.top_features])
else:
    st.info("Set project parameters on the left and click **Predict Risk**.")
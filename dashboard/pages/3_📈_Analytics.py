"""Analytics page - Dataset exploration and insights."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from code.utils.risk_levels import RISK_LEVELS, RISK_COLOURS
from code.utils.config import RAW_DIR, ROOT
from code.data_prep.load_data import load_raw
from code.data_prep.clean_data import clean

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
st.title("📈 Dataset Analytics & Insights")

# Try to load from data/raw first, then fall back to root
raw_path = RAW_DIR / "project_risk_raw_dataset.csv"
if not raw_path.exists():
    raw_path = ROOT / "project_risk_raw_dataset.csv"

df = clean(load_raw(raw_path))

# Filters
st.sidebar.header("Filters")
ptype = st.sidebar.multiselect("Project Type", sorted(df["Project_Type"].unique().tolist()))
phase = st.sidebar.multiselect("Project Phase", sorted(df["Project_Phase"].unique().tolist()))
method = st.sidebar.multiselect("Methodology", sorted(df["Methodology_Used"].unique().tolist()))

if ptype:
    df = df[df["Project_Type"].isin(ptype)]
if phase:
    df = df[df["Project_Phase"].isin(phase)]
if method:
    df = df[df["Methodology_Used"].isin(method)]

st.subheader("Risk Distribution")
counts = df["Risk_Level"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int)
st.bar_chart(counts)

st.subheader("Risk × Project Type")
ct = pd.crosstab(df["Project_Type"], df["Risk_Level"]).reindex(columns=RISK_LEVELS, fill_value=0)
st.dataframe(ct)

st.subheader("Feature Correlation (Numeric + Target)")
num_cols = [
    "Complexity_Score", "Stakeholder_Engagement_Level", "Resource_Availability",
    "Team_Turnover_Rate", "Budget_Utilization_Rate", "Communication_Frequency",
    "Schedule_Pressure", "Vendor_Reliability_Score", "Historical_Risk_Incidents",
]
# Map Risk_Level to numeric for correlation
df_corr = df[num_cols + ["Risk_Level"]].copy()
risk_map = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
df_corr["Risk_Level"] = df_corr["Risk_Level"].map(risk_map)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Numeric Feature Distributions by Risk Level")
feature_to_plot = st.selectbox("Select feature", num_cols)
fig, ax = plt.subplots(figsize=(8, 4))
for level in RISK_LEVELS:
    subset = df[df["Risk_Level"] == level][feature_to_plot].dropna()
    sns.kdeplot(subset, label=level, color=RISK_COLOURS[level], ax=ax)
ax.legend()
st.pyplot(fig)
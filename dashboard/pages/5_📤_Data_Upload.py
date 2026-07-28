"""Data Upload page - Upload custom data for analysis and prediction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from code.utils.config import FEATURE_NAMES
from code.utils.risk_levels import RISK_LEVELS, RISK_COLOURS
from code.data_prep.clean_data import clean
from code.data_prep.encode_features import encode_target
from code.api.inference import predict, predict_batch
from code.api.schemas import ProjectFeatures

st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")
st.title("📤 Upload Your Data")
st.caption("Upload a CSV file with project data to perform batch analysis and predictions.")

# File upload
st.subheader("Upload Data File")
uploaded_file = st.file_uploader(
    "Choose a data file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload a CSV or Excel file with the same columns as the training dataset."
)

if uploaded_file:
    try:
        # Read uploaded file based on type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel file.")
            st.stop()
        st.success(f"File loaded successfully! Shape: {df.shape}")
        
        # Show data preview
        with st.expander("Preview Uploaded Data", expanded=True):
            st.dataframe(df.head(10))
        
        # Check required columns
        st.subheader("Data Validation")
        missing_cols = set(FEATURE_NAMES) - set(df.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        st.success("All required feature columns found!")
        
        # Clean the data
        with st.spinner("Cleaning data..."):
            df_clean = clean(df)
        
        st.success(f"Data cleaned! Shape after cleaning: {df_clean.shape}")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df_clean))
        col2.metric("Features", len(FEATURE_NAMES))
        
        # Check if Risk_Level exists for analysis
        if "Risk_Level" in df_clean.columns:
            col3.metric("Has Target", "Yes")
            st.subheader("Risk Distribution in Uploaded Data")
            counts = df_clean["Risk_Level"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int)
            st.bar_chart(counts)
        else:
            col3.metric("Has Target", "No")
            st.info("No Risk_Level column found. This appears to be prediction-only data.")
        
        # Batch prediction
        st.subheader("Batch Prediction")
        st.write("Run predictions on the uploaded data using the trained models.")
        
        col_a, col_b = st.columns(2)
        model_choice = col_a.selectbox("Select Model", ["random_forest", "logistic_regression"])
        run_prediction = col_b.button("🚀 Run Predictions", use_container_width=True)
        
        if run_prediction:
            with st.spinner("Running predictions..."):
                try:
                    # Use fast batch prediction without SHAP
                    batch_results = predict_batch(df_clean, model_name=model_choice)
                    results = []
                    for idx, result in enumerate(batch_results):
                        results.append({
                            "Row": idx,
                            "Prediction": result["prediction"],
                            **result["probabilities"]
                        })
                    results_df = pd.DataFrame(results)
                    st.success(f"Predictions completed for {len(results_df)} records!")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                    st.stop()
                
                # Show results
                st.subheader("Prediction Results")
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"predictions_{model_choice}.csv",
                    mime="text/csv"
                )
                
                # Show prediction distribution
                st.subheader("Prediction Distribution")
                pred_counts = results_df["Prediction"].value_counts()
                st.bar_chart(pred_counts)
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

else:
    st.info("Please upload a CSV file to begin analysis.")
    
    # Show expected format
    with st.expander("Expected CSV Format"):
        st.markdown("""
        Your CSV file should contain the following columns:
        
        **Required Feature Columns:**
        - Project_Type (e.g., IT, Construction, Healthcare)
        - Complexity_Score (0-10)
        - Methodology_Used (e.g., Agile, Waterfall)
        - Project_Phase (e.g., Planning, Execution)
        - Team_Experience_Level (e.g., Junior, Mixed, Senior)
        - Project_Manager_Experience (e.g., Junior PM, Mid-level PM)
        - Resource_Availability (0-1)
        - Team_Turnover_Rate (0-1)
        - Requirement_Stability (e.g., Volatile, Moderate, Stable)
        - Risk_Management_Maturity (e.g., None, Basic, Formal, Advanced)
        - Change_Control_Maturity (e.g., None, Basic, Formal, Advanced)
        - Communication_Frequency (0-10)
        - Stakeholder_Engagement_Level (0-1)
        - Schedule_Pressure (0-1)
        - Budget_Utilization_Rate (0-1.5)
        - Historical_Risk_Incidents (0-50)
        - Vendor_Reliability_Score (0-1)
        - Tech_Environment_Stability (e.g., Legacy/Unstable, Mixed, Modern/Stable)
        
        **Optional Column:**
        - Risk_Level (for comparison with predictions: Low, Medium, High, Critical)
        """)

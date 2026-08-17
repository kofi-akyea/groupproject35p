"""Project Risk Intelligence Platform - Single-page app with top navigation."""
import os
import sys
from pathlib import Path

# Suppress joblib Windows core detection warning for parallel processing
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# Add project root directory to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from code.data_prep.load_data import load_raw
from code.data_prep.clean_data import clean
from code.utils.risk_levels import RISK_LEVELS, RISK_COLOURS, RISK_ACTIONS
from code.utils.config import RAW_DIR, ROOT
from dashboard.theme import apply_theme, COLORS

# Apply custom executive CSS styling tokens
apply_theme()

# Set Streamlit page geometry and layout configuration
st.set_page_config(
    page_title="Project Risk Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Render executive top banner header
st.markdown(f"""
<div class="app-header">
    <h1>Project Risk Intelligence Platform</h1>
    <p>Explainable ML for Four-Level Project-Risk Decision Support</p>
</div>
""", unsafe_allow_html=True)

# Initialize primary navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Predictor", "Analytics", "Model Performance", "Data Upload"])

# ==================== HOME TAB ====================
with tab1:
    # Section introduction header
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 600; color: {COLORS['text_primary']};">Dashboard Overview</h2>
        <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 13px;">Four-level ordinal risk decision-support system for project managers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load and clean primary raw dataset
    raw_path = RAW_DIR / "project_risk_raw_dataset.csv"
    if not raw_path.exists():
        raw_path = ROOT / "project_risk_raw_dataset.csv"
    df = clean(load_raw(raw_path))

    # Render summary metric cards (Total Projects, Risk Levels, Mean Complexity, Critical Share)
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="app-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 11px; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 6px;">Total Projects</div>
            <div style="font-size: 28px; font-weight: 600; color: {COLORS['text_primary']};">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="app-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 11px; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 6px;">Risk Levels</div>
            <div style="font-size: 28px; font-weight: 600; color: {COLORS['text_primary']};">{df["Risk_Level"].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="app-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 11px; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 6px;">Mean Complexity</div>
            <div style="font-size: 28px; font-weight: 600; color: {COLORS['text_primary']};">{df['Complexity_Score'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="app-card" style="text-align: center; padding: 16px;">
            <div style="font-size: 11px; color: {COLORS['text_secondary']}; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 6px;">Critical Share</div>
            <div style="font-size: 28px; font-weight: 600; color: {COLORS['risk_critical']};">{(df['Risk_Level'].eq('Critical').mean() * 100):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Render risk distribution breakdown chart
    st.markdown(f"""
    <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Risk Distribution</h3>
    """, unsafe_allow_html=True)
    counts = df["Risk_Level"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int)
    st.bar_chart(counts)

    # Render recommended actions list for each ordinal risk level
    st.markdown(f"""
    <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Recommended Actions by Risk Level</h3>
    """, unsafe_allow_html=True)

    for k in RISK_LEVELS:
        st.markdown(
            f"<div style='margin-bottom: 8px; display: flex; align-items: center;'>"
            f"<span class='apple-badge badge-{k.lower()}'>{k}</span> "
            f"<span style='color: {COLORS['text_secondary']}; margin-left: 12px; font-size: 14px;'>{RISK_ACTIONS[k]}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ==================== PREDICTOR TAB ====================
with tab2:
    # Section introduction header
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 600; color: {COLORS['text_primary']};">Project Risk Predictor</h2>
        <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 13px;">Enter project parameters; the system returns a four-level risk prediction and explainable rationale.</p>
    </div>
    """, unsafe_allow_html=True)
    
    from dashboard.components import input_form, risk_gauge, probability_bars, shap_waterfall, nlg_panel
    
    # Render interactive input configuration sliders and dropdowns
    values = input_form.render()
    
    # Split layout into prediction output column and explanation panel column
    col_pred, col_nlg = st.columns([1, 1.4])
    
    # Render form submission buttons
    with st.form("predict_form", clear_on_submit=False):
        cols = st.columns(3)
        with cols[0]:
            submitted = st.form_submit_button("Predict Risk", use_container_width=True)
        with cols[1]:
            reset = st.form_submit_button("Reset", use_container_width=True)
    
    # Handle form reset trigger
    if reset:
        st.rerun()
    
    # Execute inference when user submits prediction form
    if submitted:
        from code.api.schemas import ProjectFeatures
        # Map user input values into strongly typed inference payload
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
            # Run model prediction and store result object in session state
            from code.api.inference import predict
            result = predict(ProjectFeatures(**payload), model_name="random_forest")
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    
    # Retrieve latest prediction from session state and render output visual components
    res = st.session_state.get("last_result")
    if res:
        head = res.prediction
        colour = COLORS.get(f'risk_{head.lower()}', COLORS['text_primary'])
        
        # Render left column: Risk gauge meter, class probability bars, and action card
        with col_pred:
            prob_for_head = res.probabilities.get(head, 0.0)
            risk_gauge.render(prob_for_head, head)
            probability_bars.render(res.probabilities)
            action = RISK_ACTIONS.get(head, "Review project details.")
            st.markdown(
                f"<div style='background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 16px; border-radius: 8px; text-align: center; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>"
                f"<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {COLORS['text_secondary']}; margin-bottom: 4px;'>Recommended Action</div>"
                f"<div style='font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};'>{action}</div></div>",
                unsafe_allow_html=True,
            )
        
        # Render right column: Natural language explanation and SHAP waterfall chart
        with col_nlg:
            nlg_panel.render(res.narrative, res.top_features, colour)
            shap_waterfall.render([(f["feature"], f["shap"]) for f in res.top_features])
    else:
        # Render placeholder card before user triggers prediction
        st.markdown(f"""
        <div class="app-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 16px; color: {COLORS['text_primary']}; font-weight: 600; margin-bottom: 4px;">Ready to Predict</div>
            <div style="font-size: 13px; color: {COLORS['text_secondary']};">Set project parameters on the left and click <strong>Predict Risk</strong>.</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== ANALYTICS TAB ====================
with tab3:
    # Section introduction header
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 600; color: {COLORS['text_primary']};">Dataset Analytics & Insights</h2>
        <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 13px;">Explore the dataset with interactive filters and visualizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Load raw dataset for exploratory visualization
    raw_path = RAW_DIR / "project_risk_raw_dataset.csv"
    if not raw_path.exists():
        raw_path = ROOT / "project_risk_raw_dataset.csv"
    df = clean(load_raw(raw_path))
    
    # Render interactive multi-select filtering widgets
    col1, col2, col3 = st.columns(3)
    with col1:
        ptype = st.multiselect("Project Type", sorted(df["Project_Type"].unique().tolist()))
    with col2:
        phase = st.multiselect("Project Phase", sorted(df["Project_Phase"].unique().tolist()))
    with col3:
        method = st.multiselect("Methodology", sorted(df["Methodology_Used"].unique().tolist()))
    
    # Filter dataset according to user selections
    if ptype:
        df = df[df["Project_Type"].isin(ptype)]
    if phase:
        df = df[df["Project_Phase"].isin(phase)]
    if method:
        df = df[df["Methodology_Used"].isin(method)]
    
    # Render risk distribution bar chart
    st.markdown(f"""
    <h3 style="margin: 20px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Risk Distribution</h3>
    """, unsafe_allow_html=True)
    counts = df["Risk_Level"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int)
    st.bar_chart(counts)
    
    # Render crosstab table for risk level vs project type
    st.markdown(f"""
    <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Risk x Project Type Breakdown</h3>
    """, unsafe_allow_html=True)
    ct = pd.crosstab(df["Project_Type"], df["Risk_Level"]).reindex(columns=RISK_LEVELS, fill_value=0)
    st.dataframe(ct, use_container_width=True)
    
    # Render numeric feature correlation heatmap
    st.markdown(f"""
    <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Feature Correlation Heatmap</h3>
    """, unsafe_allow_html=True)
    num_cols = [
        "Complexity_Score", "Stakeholder_Engagement_Level", "Resource_Availability",
        "Team_Turnover_Rate", "Budget_Utilization_Rate", "Communication_Frequency",
        "Schedule_Pressure", "Vendor_Reliability_Score", "Historical_Risk_Incidents",
    ]
    df_corr = df[num_cols + ["Risk_Level"]].copy()
    risk_map = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
    df_corr["Risk_Level"] = df_corr["Risk_Level"].map(risk_map)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="Blues", center=0, ax=ax, cbar_kws={'shrink': 0.8})
    plt.xticks(fontsize=8, rotation=35, ha="right")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Render density distribution plots for individual features split by risk level
    st.markdown(f"""
    <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Numeric Feature Distributions by Risk Level</h3>
    """, unsafe_allow_html=True)
    feature_to_plot = st.selectbox("Select Feature to Plot", num_cols)
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    for level in RISK_LEVELS:
        subset = df[df["Risk_Level"] == level][feature_to_plot].dropna()
        sns.kdeplot(subset, label=level, color=RISK_COLOURS[level], ax=ax)
    ax.legend(fontsize=8)
    ax.set_xlabel(feature_to_plot, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

# ==================== MODEL PERFORMANCE TAB ====================
with tab4:
    # Section introduction header
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 600; color: {COLORS['text_primary']};">Model Performance & Comparison</h2>
        <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 13px;">Ordinal-aware evaluation on the held-out test set.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        import json
        import joblib
        import numpy as np
        from sklearn.metrics import roc_curve, auc
        from code.utils.config import PROCESSED_DIR, MODELS_DIR
        from code.models.ordinal_metrics import ordinal_report
        from code.data_prep.load_data import load_processed
        
        # Verify test dataset file existence
        x_test_path = PROCESSED_DIR / "X_test.parquet"
        y_test_path = PROCESSED_DIR / "y_test.parquet"
        
        if not x_test_path.exists() or not y_test_path.exists():
            st.error(f"Test data not found. Expected files:\n- {x_test_path}\n- {y_test_path}\n\nPlease run the pipeline first to generate test data.")
        else:
            # Load held-out test feature vectors and labels
            X_test = load_processed("X_test")
            y_test = load_processed("y_test")["Risk_Level"]
            
            # List supported model candidates
            model_names = ["logistic_regression", "random_forest", "ordinal_logistic_regression", "xgboost", "svm_rbf", "knn"]
            model_labels = {
                "logistic_regression": "Logistic Regression",
                "random_forest": "Random Forest",
                "ordinal_logistic_regression": "Ordinal Logistic Regression",
                "xgboost": "XGBoost",
                "svm_rbf": "SVM (RBF Kernel)",
                "knn": "K-Nearest Neighbors"
            }
            results = {}
            
            # Load each trained model pipeline and compute ordinal evaluation report
            for name in model_names:
                model_path = MODELS_DIR / f"{name}.joblib"
                if model_path.exists():
                    try:
                        pipe = joblib.load(model_path)
                        proba = pipe.predict_proba(X_test)
                        pred = proba.argmax(axis=1)
                        rep = ordinal_report(y_test, pred, proba, RISK_LEVELS)
                        results[name] = rep
                    except Exception as e:
                        st.warning(f"Error loading {name}: {e}")
                else:
                    st.warning(f"Model not found: {model_path}")
            
            # Render model performance summary table
            if results:
                st.markdown(f"""
                <h3 style="margin: 16px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Key Metrics Comparison</h3>
                """, unsafe_allow_html=True)
                
                metrics_data = []
                for name in model_names:
                    if name in results:
                        metrics_data.append({
                            "Model": model_labels[name],
                            "Accuracy": f"{results[name]['accuracy']:.4f}",
                            "Within-One": f"{results[name]['within_one']:.4f}",
                            "QWK": f"{results[name]['qwk']:.4f}",
                            "Macro F2": f"{results[name]['macro_f2']:.4f}"
                        })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
            
                # Render normalized confusion matrix heatmap for top model
                best_model = "random_forest"
                if best_model in results:
                    st.markdown(f"""
                    <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Confusion Matrix - {model_labels[best_model]} (Normalised)</h3>
                    """, unsafe_allow_html=True)
                    cm = np.array(results[best_model]["confusion_matrix"])
                    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                    fig, ax = plt.subplots(figsize=(5, 3.8), dpi=100)
                    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=RISK_LEVELS,
                                yticklabels=RISK_LEVELS, cmap="Blues", ax=ax, cbar=False)
                    ax.set_xlabel("Predicted Class", fontsize=9)
                    ax.set_ylabel("True Class", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
            
                # Render One-vs-Rest macro ROC curves for candidate models
                st.markdown(f"""
                <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Macro ROC Curves (One-vs-Rest)</h3>
                """, unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                for name in model_names:
                    if name in results:
                        pipe = joblib.load(MODELS_DIR / f"{name}.joblib")
                        proba = pipe.predict_proba(X_test)
                        for k, cls in enumerate(RISK_LEVELS):
                            fpr, tpr, _ = roc_curve((y_test == k).astype(int), proba[:, k])
                            ax.plot(fpr, tpr, color=RISK_COLOURS[cls],
                                    label=f"{model_labels[name]} - {cls} (AUC={auc(fpr, tpr):.3f})",
                                    linestyle="--" if "logistic" in name else "-", linewidth=1.2)
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                ax.legend(fontsize=7, loc="lower right")
                ax.set_xlabel("False Positive Rate", fontsize=9)
                ax.set_ylabel("True Positive Rate", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
            
                # Render detailed per-class metrics table (Precision, Recall, F1, F2)
                st.markdown(f"""
                <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Per-Class Performance - {model_labels[best_model]}</h3>
                """, unsafe_allow_html=True)
                if "per_class_report" in results.get(best_model, {}):
                    per_class_df = pd.DataFrame(results[best_model]["per_class_report"]).T
                    st.dataframe(per_class_df.style.format("{:.3f}"), use_container_width=True)
            else:
                st.warning("No trained models found. Run the pipeline first.")
    except Exception as e:
        st.error(f"Error in Model Performance tab: {e}")
        import traceback
        traceback.print_exc()

# ==================== DATA UPLOAD TAB ====================
with tab5:
    # Section introduction header
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 600; color: {COLORS['text_primary']};">Upload Your Data</h2>
        <p style="margin: 0; color: {COLORS['text_secondary']}; font-size: 13px;">Upload a CSV file with project data to perform batch analysis and predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from code.utils.config import FEATURE_NAMES
        from code.data_prep.encode_features import encode_target
        from code.api.inference import predict, predict_batch
        from code.api.schemas import ProjectFeatures
        
        # Display expected CSV schema details inside expander widget
        with st.expander("Expected CSV Format", expanded=True):
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
        
        # Render file upload drag-and-drop widget
        st.markdown(f"""
        <h3 style="margin: 20px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Upload Data File</h3>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file with the same columns as the training dataset."
        )
        
        # Parse and process uploaded data file
        if uploaded_file:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file type. Please upload CSV or Excel file.")
                    st.stop()
                st.success(f"File loaded successfully! Shape: {df.shape}")
                
                # Render data preview dataframe
                with st.expander("Preview Uploaded Data", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Validate required feature columns exist in uploaded dataframe
                st.markdown(f"""
                <h3 style="margin: 20px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Data Validation</h3>
                """, unsafe_allow_html=True)
                missing_cols = set(FEATURE_NAMES) - set(df.columns)
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    st.stop()
                
                st.success("All required feature columns found!")
                
                # Execute automated cleaning on uploaded data
                with st.spinner("Cleaning data..."):
                    df_clean = clean(df)
                
                st.success(f"Data cleaned! Shape after cleaning: {df_clean.shape}")
                
                # Render dataset overview metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Records", len(df_clean))
                col2.metric("Features", len(FEATURE_NAMES))
                
                # Display target column risk distribution if included in uploaded file
                if "Risk_Level" in df_clean.columns:
                    col3.metric("Has Target", "Yes")
                    st.markdown(f"""
                    <h3 style="margin: 20px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Risk Distribution in Uploaded Data</h3>
                    """, unsafe_allow_html=True)
                    counts = df_clean["Risk_Level"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int)
                    st.bar_chart(counts)
                else:
                    col3.metric("Has Target", "No")
                    st.info("No Risk_Level column found. This appears to be prediction-only data.")
                
                # Batch prediction configuration controls
                st.markdown(f"""
                <h3 style="margin: 24px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Batch Prediction</h3>
                """, unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                model_choice = col_a.selectbox("Select Model", ["ordinal_logistic_regression", "random_forest", "logistic_regression", "xgboost", "svm_rbf", "knn"])
                run_prediction = col_b.button("Run Predictions", use_container_width=True)
                
                # Execute batch predictions on all uploaded dataset rows
                if run_prediction:
                    with st.spinner("Running predictions..."):
                        try:
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
                            import traceback
                            traceback.print_exc()
                            st.stop()
                    
                    # Display batch prediction output table
                    st.markdown(f"""
                    <h3 style="margin: 20px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Prediction Results</h3>
                    """, unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Provide CSV download button for prediction output
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"predictions_{model_choice}.csv",
                        mime="text/csv"
                    )
                    
                    # Display batch prediction distribution chart
                    st.markdown(f"""
                    <h3 style="margin: 20px 0 12px 0; font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};">Prediction Distribution</h3>
                    """, unsafe_allow_html=True)
                    pred_counts = results_df["Prediction"].value_counts()
                    st.bar_chart(pred_counts)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
                import traceback
                traceback.print_exc()
        else:
            st.info("Please upload a CSV file to begin analysis.")
    except Exception as e:
        st.error(f"Error in Data Upload tab: {e}")
        import traceback
        traceback.print_exc()
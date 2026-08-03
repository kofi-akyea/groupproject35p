# Figures and Tables for Dissertation

This document contains all figures and tables referenced in the dissertation chapters, formatted in markdown for inclusion in the final document.

## Chapter 3: Research Design

### Table 3.1: Risk Level Distribution in Dataset

| Risk Level | Proportion |
|-----------|------------|
| Low | 20.15% |
| Medium | 34.90% |
| High | 25.90% |
| Critical | 19.05% |

### Table 3.2: Stakeholder Engagement Level Transformation Mapping

| Stakeholder Engagement | Numerical Value |
|----------------------|-----------------|
| Poor | 0.00 |
| Low | 0.25 |
| Medium | 0.50 |
| High | 0.75 |
| Excellent | 1.00 |

### Table 3.3: Dataset Partitioning

| Dataset | Records | Proportion |
|---------|---------|------------|
| Training | 2,800 | 70% |
| Validation | 600 | 15% |
| Testing | 600 | 15% |
| Total | 4,000 | 100% |

### Table 3.4: Final Feature Set (18 Predictors)

**Project Characteristics:**
- Project_Type (nominal): Construction, IT, Healthcare, Manufacturing, R&D, Marketing
- Complexity_Score (numeric): 0.0-10.0
- Methodology_Used (nominal): Waterfall, Agile, Scrum, Kanban, Hybrid
- Project_Phase (ordinal): Initiation, Planning, Execution, Monitoring, Closure

**Team Factors:**
- Team_Experience_Level (ordinal): Junior, Mixed, Senior, Expert
- Project_Manager_Experience (ordinal): Junior PM, Mid-level PM, Senior PM, Certified PM
- Resource_Availability (numeric): 0.0-1.0
- Team_Turnover_Rate (numeric): 0.0-1.0

**Governance:**
- Requirement_Stability (ordinal): Volatile, Moderate, Stable
- Risk_Management_Maturity (ordinal): None, Basic, Formal, Advanced
- Change_Control_Maturity (ordinal): None, Basic, Formal, Advanced
- Communication_Frequency (numeric): 0.0-10.0
- Stakeholder_Engagement_Level (numeric): 0.0-1.0 (transformed from categorical)

**Schedule/Budget/External:**
- Schedule_Pressure (numeric): 0.0-1.0
- Budget_Utilization_Rate (numeric): 0.0-1.5
- Historical_Risk_Incidents (numeric): 0-20
- Vendor_Reliability_Score (numeric): 0.0-1.0
- Tech_Environment_Stability (ordinal): Legacy/Unstable, Mixed, Modern/Stable

---

## Chapter 4: Implementation

### Figure 4.1: Project Directory Structure

```
DSS SYSTEM GROUP/
├── pipeline.py                        # Main training pipeline (run first)
├── requirements.txt                   # Python dependencies
├── project_risk_raw_dataset.csv       # Raw dataset (4,000 records)
│
├── data/                              # Data storage
│   ├── raw/
│   │   └── project_risk_raw_dataset.csv
│   └── processed/
│       ├── X_train.parquet
│       ├── X_val.parquet
│       ├── X_test.parquet
│       ├── y_train.parquet
│       ├── y_val.parquet
│       ├── y_test.parquet
│       └── class_distribution.csv
│
├── models/                            # Trained model files
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── model_card.json
│
├── reports/                           # Evaluation reports
│   ├── logistic_regression_test_report.json
│   └── random_forest_test_report.json
│
├── dashboard/                         # Streamlit web interface
│   ├── app.py                         # Main entry point (multi-page)
│   ├── pages/                         # Five dashboard pages
│   │   ├── 1_🏠_Home.py
│   │   ├── 2_🔮_Predictor.py
│   │   ├── 3_📈_Analytics.py
│   │   ├── 4_⚖️_Model_Performance.py
│   │   └── 5_📤_Data_Upload.py
│   └── components/                    # Reusable UI widgets
│       ├── input_form.py
│       ├── risk_gauge.py
│       ├── probability_bars.py
│       ├── shap_waterfall.py
│       └── nlg_panel.py
│
├── code/                              # Source code modules
│   ├── data_prep/                     # Data preprocessing
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   ├── clean_data.py
│   │   ├── encode_features.py
│   │   └── split_data.py
│   ├── models/                        # Model training & evaluation
│   │   ├── train_logreg.py
│   │   ├── train_rf.py
│   │   ├── evaluate.py
│   │   └── ordinal_metrics.py
│   ├── explainability/                # Model explanations
│   │   ├── shap_engine.py
│   │   └── narrative_builder.py
│   ├── api/                           # Flask REST API
│   │   ├── app.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   ├── inference.py
│   │   └── errors.py
│   └── utils/                         # Utilities
│       ├── config.py
│       └── risk_levels.py
│
└── README.md                          # Project documentation
```

### Figure 4.2: Deployment Architecture and Data Flow

```
User Browser → Streamlit (port 8501) → Flask API (port 5000) → models/ and code/ artefacts
```

**Architecture Description:**
- **Presentation Layer:** Streamlit dashboard (multi-page web interface)
- **Application Logic Layer:** Flask REST API with `/predict`, `/explain`, `/health` endpoints
- **Data Layer:** Serialised preprocessing pipelines, trained models, and processed datasets

### Figure 4.3: Streamlit Predictor Page - Example Prediction

*(Insert dashboard screenshot showing:)*
- Input form with 18 feature controls on the left
- Risk gauge showing predicted risk level in the centre
- Probability bars for all four risk classes
- SHAP waterfall chart showing top feature contributions
- Natural language explanation panel with actionable insights

---

## Chapter 5: Testing

### Table 5.1: Performance Metrics (API Response Times)

| Metric | /predict | /explain |
|--------|----------|----------|
| Average latency (ms) | ~180 | ~420 |
| Maximum latency (ms) | ~340 | ~890 |
| Requests within 2s target | 100% | 100% |

### Table 5.2: Predictive Performance on Test Set (600 Samples)

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Overall Accuracy | 0.502 | 0.507 |
| Macro averaged F1 score | 0.508 | 0.508 |
| Quadratic weighted Cohen's Kappa | 0.666 | 0.622 |
| Within-one accuracy | 0.905 | 0.927 |
| Macro AUC (OVR) | 0.781 | 0.775 |

### Table 5.3: Logistic Regression Confusion Matrix

```
          Predicted
Actual    Low  Medium  High  Critical
Low       91     21     9        0
Medium    58     78    49       25
High      14     38    52       51
Critical   0      9    25       80
```

### Table 5.4: Random Forest Confusion Matrix

```
          Predicted
Actual    Low  Medium  High  Critical
Low       66     49     6        0
Medium    19    134    45       12
High       1     71    45       38
Critical   2     23    30       59
```

### Table 5.5: Per-Class Performance (Random Forest)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.750 | 0.545 | 0.632 | 121 |
| Medium | 0.484 | 0.638 | 0.550 | 210 |
| High | 0.357 | 0.290 | 0.320 | 155 |
| Critical | 0.541 | 0.518 | 0.529 | 114 |
| Macro avg | 0.533 | 0.498 | 0.508 | 600 |

---

## Chapter 6: Presentation of Results

### Figure 6.1: Comparative Performance of Logistic Regression and Random Forest on Test Set

*(Insert grouped bar chart with the following actual values)*

**Logistic Regression:**
- Accuracy: 0.502
- Macro F1: 0.508
- QWK: 0.666

**Random Forest:**
- Accuracy: 0.507
- Macro F1: 0.508
- QWK: 0.622

### Table 6.1: Final Feature Set (18 Features)

| Feature | Type | Range/Values | Group |
|---------|------|--------------|-------|
| Project_Type | Nominal | Construction, IT, Healthcare, Manufacturing, R&D, Marketing | Project |
| Complexity_Score | Numeric | 0.0-10.0 | Project |
| Methodology_Used | Nominal | Waterfall, Agile, Scrum, Kanban, Hybrid | Project |
| Project_Phase | Ordinal | Initiation, Planning, Execution, Monitoring, Closure | Project |
| Team_Experience_Level | Ordinal | Junior, Mixed, Senior, Expert | Team |
| Project_Manager_Experience | Ordinal | Junior PM, Mid-level PM, Senior PM, Certified PM | Team |
| Resource_Availability | Numeric | 0.0-1.0 | Team |
| Team_Turnover_Rate | Numeric | 0.0-1.0 | Team |
| Requirement_Stability | Ordinal | Volatile, Moderate, Stable | Governance |
| Risk_Management_Maturity | Ordinal | None, Basic, Formal, Advanced | Governance |
| Change_Control_Maturity | Ordinal | None, Basic, Formal, Advanced | Governance |
| Communication_Frequency | Numeric | 0.0-10.0 | Governance |
| Stakeholder_Engagement_Level | Numeric | 0.0-1.0 | Governance |
| Schedule_Pressure | Numeric | 0.0-1.0 | Schedule/Budget/External |
| Budget_Utilization_Rate | Numeric | 0.0-1.5 | Schedule/Budget/External |
| Historical_Risk_Incidents | Numeric | 0-20 | Schedule/Budget/External |
| Vendor_Reliability_Score | Numeric | 0.0-1.0 | Schedule/Budget/External |
| Tech_Environment_Stability | Ordinal | Legacy/Unstable, Mixed, Modern/Stable | Schedule/Budget/External |

### Figure 6.2: Streamlit Predictor Page - Example Prediction

*(Insert dashboard screenshot showing:)*
- Input form with 18 feature controls
- Risk gauge showing predicted risk level
- Probability bars for all four risk classes
- SHAP waterfall chart showing top feature contributions
- Natural language explanation panel

---

## Dashboard Page Screenshots

### Figure: Home Page
*(Insert screenshot of Home page showing:)*
- Project risk platform overview
- Key metrics (total projects, risk distribution)
- Risk distribution bar chart
- Recommended actions with color coding

### Figure: Analytics Page
*(Insert screenshot of Analytics page showing:)*
- Data loading and filtering controls
- Risk distribution charts
- Crosstab analysis
- Correlation heatmap
- Numeric feature distributions by risk level

### Figure: Model Performance Page
*(Insert screenshot of Model Performance page showing:)*
- Model comparison metrics
- Confusion matrices
- Feature importance rankings
- Performance over time charts

### Figure: Data Upload Page
*(Insert screenshot of Data Upload page showing:)*
- File upload interface (CSV, Excel support)
- Data cleaning summary
- Batch prediction results
- Download functionality for processed data

---

## Code Excerpts

### Code Excerpt 4.1: Pipeline Assembly in pipeline.py

```python
from code.data_prep.load_data import load_raw
from code.data_prep.validate_data import validate_data
from code.data_prep.clean_data import clean
from code.data_prep.encode_features import encode_target
from code.data_prep.split_data import split
from code.utils.config import FEATURE_NAMES

# 1. Load and validate
df = load_raw(raw_csv)
errors = validate_data(df)

# 2. Clean
df = clean(df)

# 3. Prepare features and target
X = df[FEATURE_NAMES].copy()
y = encode_target(df["Risk_Level"])

# 4. Split
X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
```

### Code Excerpt 4.2: Random Forest Training in code/models/train_rf.py

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/random_forest.joblib')
    return rf
```

### Code Excerpt 4.3: SHAP Engine for Pipeline Predictions (shap_engine.py)

```python
def shap_for_pipeline(pipeline, X_row: pd.DataFrame) -> dict:
    """Return per-class probabilities and per-feature SHAP values."""
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    Xt = pre.transform(X_row)

    if hasattr(clf, "estimators_"):
        # Tree-based model (Random Forest)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        origins = build_feature_origin_map(pre)
        probs = clf.predict_proba(Xt)[0]
        per_class = {CLASS_NAMES[k]: float(probs[k]) for k in range(len(CLASS_NAMES))}
        per_class_shap = {}
        for k in range(len(CLASS_NAMES)):
            # Handle empty SHAP values
            if isinstance(sv, list):
                shap_row = sv[k][0] if len(sv) > k and len(sv[k]) > 0 else np.zeros(len(origins))
            else:
                shap_row = sv[0] if len(sv) > 0 else np.zeros(len(origins))
            per_class_shap[CLASS_NAMES[k]] = _aggregate_origin(shap_row, origins)
        head_idx = int(np.argmax(probs))
        head = CLASS_NAMES[head_idx]
        return {
            "probabilities": per_class,
            "head": head,
            "head_shap": per_class_shap[head],
            "head_class_index": head_idx,
        }
```

### Code Excerpt 4.4: API Endpoint Definitions (code/api/routes.py)

```python
from flask import Blueprint, request, jsonify
from code.api.schemas import ProjectFeatures
from code.api.inference import predict
from code.api.errors import error_response, validate_payload

bp = Blueprint("api", __name__)

@bp.route("/predict", methods=["POST"])
def route_predict():
    """Predict project risk level from input features."""
    payload = request.get_json(silent=True) or {}
    err = validate_payload(payload)
    if err:
        return error_response(err, 400)
    features = ProjectFeatures(**payload)
    result = predict(features, model_name=payload.get("model", "random_forest"))
    return jsonify({
        "request_id": result.request_id,
        "prediction": result.prediction,
        "probabilities": result.probabilities,
        "shap": result.shap,
        "top_features": result.top_features,
        "narrative": result.narrative,
        "model_version": result.model_version,
    })

@bp.route("/explain", methods=["POST"])
def route_explain():
    """Alias for /predict - reserved for explainability-only clients."""
    return route_predict()

@bp.route("/health", methods=["GET"])
def route_health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "version": "1.0.0",
    })
```

### Code Excerpt 4.5: Predictor Page Logic (dashboard/pages/2_🔮_Predictor.py)

```python
import streamlit as st
from code.api.inference import predict
from code.api.schemas import ProjectFeatures
from dashboard.components import input_form, risk_gauge, probability_bars, shap_waterfall, nlg_panel

values = input_form.render()

if st.form_submit_button("🚀 Predict Risk"):
    # Ensure all values are properly typed
    payload = {
        "Project_Type": str(values["Project_Type"]),
        "Complexity_Score": float(values["Complexity_Score"]),
        # ... remaining 16 features
    }
    result = predict(ProjectFeatures(**payload), model_name="random_forest")
    
    # Display results
    risk_gauge.render(result.probabilities.get(result.prediction, 0.0), result.prediction)
    probability_bars.render(result.probabilities)
    nlg_panel.render(result.narrative, result.top_features, RISK_COLOURS.get(result.prediction, "#888"))
    shap_waterfall.render([(f["feature"], f["shap"]) for f in result.top_features])
```

### Code Excerpt 4.6: Batch Prediction Implementation

```python
def predict_batch(df: pd.DataFrame, model_name: str = DEFAULT_MODEL) -> list:
    """Run batch predictions without SHAP for faster processing."""
    pipe = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    
    # Ensure categorical columns are strings
    categorical_cols = [
        "Project_Type", "Methodology_Used", "Project_Phase",
        "Team_Experience_Level", "Project_Manager_Experience",
        "Requirement_Stability", "Risk_Management_Maturity", 
        "Change_Control_Maturity", "Tech_Environment_Stability"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Transform and predict in batch
    Xt = pre.transform(df)
    probs = clf.predict_proba(Xt)
    preds = clf.predict(Xt)
    
    # Convert to class names
    from code.utils.risk_levels import RISK_LEVELS
    results = []
    for i in range(len(df)):
        pred_class = RISK_LEVELS[preds[i]]
        prob_dict = {RISK_LEVELS[k]: float(probs[i][k]) for k in range(len(RISK_LEVELS))}
        results.append({
            "prediction": pred_class,
            "probabilities": prob_dict
        })
    
    return results
```

---

## Notes for Figure Creation

1. **Dashboard Screenshots:** These should be captured from the actual running Streamlit application at `http://localhost:8501` or the configured port.

2. **Performance Bar Chart (Figure 6.1):** Create using matplotlib, plotly, or Excel with the actual values provided:
   - X-axis: Metrics (Accuracy, Macro F1, QWK)
   - Y-axis: Value (0.0-1.0)
   - Grouped bars for Logistic Regression and Random Forest

3. **Confusion Matrices:** Can be visualized as heatmaps using seaborn or matplotlib with the actual values provided.

4. **Architecture Diagrams:** The text-based representations can be converted to proper diagrams using:
   - Draw.io
   - Lucidchart
   - Mermaid.js
   - PowerPoint/Visio

All values in this document are derived from the actual project implementation and evaluation reports, ensuring accuracy and alignment with the codebase.

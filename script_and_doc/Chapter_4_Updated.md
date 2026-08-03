# Chapter Four: Implementation

## 4.1 Introduction

This chapter documents the technical construction of the web-based intelligent Decision Support System (DSS) artefact. It translates the research design from Chapter Three into a functioning software system. The implementation was partitioned across three specialist roles: data engineering, model development and explainability, and web application integration. This chapter first outlines the project structure and development environment, then presents the three-tier architectural implementation in detail, provides walkthroughs of key code modules with interpretive commentary, and concludes with a synthesis of the major design decisions and their trade-offs.

## 4.2 Project Structure and Development Environment

All source code was maintained in a Git repository with the directory hierarchy shown in Figure 4.1. The environment was defined by a requirements.txt file at the project root, capturing exact package versions. The core technology stack consisted of Python 3.10, pandas, NumPy, scikit-learn 1.2, SHAP 0.41, Flask 2.3, and Streamlit 1.25.

### Figure 4.1: Project directory structure

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

**Interpretation:** The root contains the master training script `pipeline.py`, which orchestrates the entire data preparation and model training workflow. The `code/` directory is organised by functional domain: preprocessing, modelling, explainability, API, and utilities. The `dashboard/` folder holds the Streamlit application with five pages, while `models/` and `reports/` store serialised artefacts. This layout cleanly separates development from operational artefacts and supports independent testing of each component.

All development was performed in a Python virtual environment. The requirements.txt listed libraries with pinned versions (e.g., scikit-learn==1.2.0, shap==0.41.0), ensuring that the environment could be recreated identically.

## 4.3 Architectural Overview

The DSS implements a three-tier architecture. Figure 4.2 shows the deployment view.

- **Data Layer:** The `data/` folder holds raw and processed datasets. Serialised preprocessing objects and trained models reside in `models/`.
- **Application Logic Layer:** The Flask REST API in `code/api/` loads the serialised artefacts and serves three endpoints (`/predict`, `/explain`, `/health`). The inference logic in `code/api/inference.py` calls the preprocessing pipeline and the model.
- **Presentation Layer:** The Streamlit dashboard (`dashboard/app.py`) provides an interactive user interface. It communicates with the Flask API over HTTP, sending JSON requests and rendering the responses.

### Figure 4.2: Deployment architecture and data flow

```
User Browser → Streamlit (port 8501) → Flask API (port 5000) → models/ and code/ artefacts
```

**Interpretation:** The user interacts solely with the Streamlit interface. Compute-intensive prediction and explanation generation occur in the Flask backend. This decoupling isolates the frontend from the ML runtime, so models can be updated without modifying the dashboard, and the API can be tested independently.

## 4.4 Data Preprocessing Implementation

The data preprocessing stage follows a modular design within `code/data_prep/`. The master script `pipeline.py` calls each module in sequence, producing a clean, encoded, and split dataset ready for training.

### Code Excerpt 4.1: Pipeline assembly in pipeline.py

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

**Commentary:** The pipeline enforces a strict sequence. The `clean()` function in `clean_data.py` handles missing values, converts categorical variables (including special handling for `Stakeholder_Engagement_Level`), and clips numeric values to valid ranges. The configuration file `config.py` serves as the single source of truth for feature specifications, ensuring consistency between training and inference.

### Stakeholder_Engagement_Level Handling

A critical preprocessing step is the conversion of `Stakeholder_Engagement_Level` from categorical to numeric:

```python
SE_LEVEL_MAP = {"Low": 0.25, "Medium": 0.50, "High": 0.75, "Excellent": 1.0, "Poor": 0.0}

# In clean_data.py
if "Stakeholder_Engagement_Level" in df.columns:
    df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].astype(str)
    df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].map(SE_LEVEL_MAP).fillna(0.5)
    df["Stakeholder_Engagement_Level"] = pd.to_numeric(df["Stakeholder_Engagement_Level"], errors="coerce")
```

**Commentary:** This conversion ensures that the stakeholder engagement feature is properly numeric before model training, preventing type conversion errors during inference.

## 4.5 Machine Learning Model Development and Explainability Integration

Model training is performed by the modules in `code/models/`. The training pipeline is called from `pipeline.py` after preprocessing.

### Code Excerpt 4.2: Random Forest training in code/models/train_rf.py

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

**Commentary:** `class_weight='balanced'` adjusts for the moderate under-representation of the Critical class. Random Forest was selected as the primary inference model, while Logistic Regression served as a transparent baseline for comparison.

### Explainability Integration

The explainability module in `code/explainability/` consists of two files. `shap_engine.py` initialises a TreeExplainer for the Random Forest and computes both global and local SHAP values. `narrative_builder.py` converts the raw SHAP values into structured JSON and natural language strings.

### Code Excerpt 4.3: SHAP engine for pipeline predictions (shap_engine.py)

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

**Commentary:** TreeExplainer computes exact SHAP values without sampling, guaranteeing deterministic and fast explanations. The function includes error handling for empty SHAP values to prevent index out of bounds errors during batch predictions.

## 4.6 Web Application Integration

The Flask API in `code/api/` is the integration backbone. It loads the preprocessing pipeline, the trained Random Forest model, and the SHAP explainer once at startup. Three endpoints are exposed: `/predict`, `/explain`, and `/health`, all defined in `routes.py` and supported by `schemas.py` for input validation.

### Code Excerpt 4.4: API endpoint definitions (code/api/routes.py)

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

**Commentary:** The API uses a Blueprint to allow flexible routing. Input validation is enforced by the `validate_payload()` function, which checks for required fields, data types, and value ranges. The `/predict` endpoint returns both prediction and explanation in a single response, reducing API calls for the dashboard. The `/explain` endpoint is maintained as an alias for compatibility.

The Streamlit dashboard (`dashboard/app.py`) is a multi-page application. The entry point uses Streamlit's pages/ directory to present separate views: Home, Predictor, Analytics, Model Performance, and Data Upload. The Predictor page (Figure 4.3) contains the interactive risk prediction interface.

### Code Excerpt 4.5: Predictor page logic (dashboard/pages/2_🔮_Predictor.py)

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

**Commentary:** The form collects all 18 features using the `input_form` component. Upon submission, the dashboard calls the `predict()` function directly (bypassing HTTP for performance in local deployment). The predicted class is displayed as a risk gauge, probabilities as horizontal bars, and the top SHAP contributions as a waterfall chart and natural language explanation.

### Figure 4.3: Screenshot of the Predictor page

*(Insert dashboard screenshot with form on left, risk gauge and probability bars in centre, explanation below)*

**Interpretation:** The dashboard communicates the model's reasoning transparently. The colour-coded risk gauge, clear probability visualisation, and explanation bullets enable the user to link specific project parameters to the predicted risk outcome.

## 4.7 Data Upload Feature

A new addition to the dashboard is the Data Upload page (`dashboard/pages/5_📤_Data_Upload.py`), which allows users to upload CSV or Excel files for batch analysis and prediction.

### Code Excerpt 4.6: Batch prediction implementation

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

**Commentary:** The batch prediction function processes all rows in a single transformation, skipping SHAP computation for performance. This enables rapid processing of large datasets uploaded by users.

## 4.8 Design Decisions and Trade-offs

Several architectural and algorithmic choices were made during implementation, each accompanied by deliberate trade-offs.

**Modular folder structure over monolithic scripts:** The project adopted a clear separation into `code/data_prep/`, `code/models/`, and `code/api/`. This increases the number of files but dramatically improves maintainability and testability.

**Flask Blueprint vs. single application file:** Using a Blueprint in `code/api/routes.py` allows the API to be registered under a URL prefix and easily extended with new endpoints without refactoring.

**Single `/predict` endpoint returning both prediction and explanation:** Rather than separate endpoints, the API returns both in a single response. This reduces HTTP overhead for the dashboard while maintaining the option for an `/explain` alias for compatibility.

**Ordinal encoding of ordered categoricals:** Features like project phase were encoded ordinally to preserve their natural order. One-hot encoding would have discarded this ordering and increased dimensionality.

**Balanced class weights over SMOTE:** `class_weight='balanced'` was used in Random Forest rather than synthetic oversampling. This avoids generating artificial data points and simplifies the pipeline.

**Batch prediction without SHAP for data upload:** The Data Upload page uses a separate `predict_batch()` function that skips SHAP computation for performance. This trade-off prioritises speed over explainability for bulk processing.

**HTML bold tags over Markdown asterisks:** In the explanation panel, HTML `<b>` tags are used instead of Markdown `**` syntax to prevent asterisks from appearing in the rendered output.

## 4.9 Chapter Summary

This chapter detailed the implementation of the three-tier intelligent DSS. A modular, reproducible preprocessing pipeline transforms raw project data using ordinal and one-hot encoding. The Random Forest model serves as the primary prediction engine, with TreeSHAP providing global and local explanations. A Flask Blueprint API exposes the models through `/predict`, `/explain`, and `/health` endpoints, and a multi-page Streamlit dashboard delivers an interactive user interface with five pages including a new Data Upload feature for batch processing. Key design decisions were explicitly justified, and their trade-offs acknowledged. The implemented artefact is ready for systematic evaluation, which is the subject of Chapter Five.

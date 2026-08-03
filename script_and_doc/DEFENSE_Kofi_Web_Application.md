# Defense Guide: Kofi Awuette Akyea - Web Application Integration

## Role Overview

**Primary Responsibility:** DSS Artefact Development and Web Application Integration
**Chapter Reference:** Chapter Four, Section 4.6 - Web Application Integration
**Key Deliverables:** Three-tier web-based DSS with interactive Streamlit dashboard

---

## 1. My Contribution Summary

### Core Responsibilities
- **Architecture Design:** Implemented three-tier architecture (data, application, presentation)
- **Flask API Development:** Built REST API with `/predict`, `/explain`, `/health` endpoints
- **Streamlit Dashboard:** Created multi-page interactive dashboard (5 pages)
- **Component Development:** Built reusable UI components (input form, risk gauge, probability bars, SHAP waterfall, NLG panel)
- **Data Upload Feature:** Implemented batch data upload and processing capability
- **Integration Testing:** Ensured seamless communication between all tiers

### Key Files Implemented
- `code/api/app.py` - Flask application factory
- `code/api/routes.py` - API endpoint definitions
- `code/api/schemas.py` - Pydantic request/response schemas
- `code/api/inference.py` - Prediction inference logic
- `code/api/errors.py` - Error handling and validation
- `dashboard/app.py` - Main Streamlit application
- `dashboard/pages/1_🏠_Home.py` - Home page with overview
- `dashboard/pages/2_🔮_Predictor.py` - Interactive predictor page
- `dashboard/pages/3_📈_Analytics.py` - Analytics page with visualisations
- `dashboard/pages/4_⚖️_Model_Performance.py` - Model performance comparison
- `dashboard/pages/5_📤_Data_Upload.py` - Batch data upload and processing
- `dashboard/components/input_form.py` - Dynamic input form component
- `dashboard/components/risk_gauge.py` - Risk gauge visualization
- `dashboard/components/probability_bars.py` - Probability bar charts
- `dashboard/components/shap_waterfall.py` - SHAP waterfall chart
- `dashboard/components/nlg_panel.py` - Natural language explanation panel

---

## 2. Technical Decisions and Rationale

### Decision 1: Three-Tier Architecture
**What I did:** Implemented separate data, application, and presentation layers
**Why:** Enables independent development, testing, and deployment; isolates ML runtime from frontend
**Evidence:** Architecture documented in Chapter 4, Figure 4.2

### Decision 2: Flask Blueprint for API
**What I did:** Used Flask Blueprint instead of single application file
**Why:** Allows flexible routing, easy extension with new endpoints, better code organisation
**Evidence:** Implemented in `code/api/routes.py` line 7

### Decision 3: Single `/predict` Endpoint
**What I did:** Combined prediction and explanation in single endpoint instead of separate calls
**Why:** Reduces HTTP overhead for dashboard, maintains `/explain` as alias for compatibility
**Evidence:** Implemented in `code/api/routes.py` line 10-27

### Decision 4: Streamlit Multi-Page Architecture
**What I did:** Used Streamlit's pages/ directory for separate dashboard pages
**Why:** Clean separation of concerns, better navigation, easier maintenance
**Evidence:** 5 pages in `dashboard/pages/` directory

### Decision 5: Data Upload with Batch Prediction
**What I did:** Added Data Upload page with separate `predict_batch()` function
**Why:** Enables processing of large datasets without SHAP overhead for performance
**Evidence:** Implemented in `dashboard/pages/5_📤_Data_Upload.py` and `code/api/inference.py`

---

## 3. Challenges Faced and Solutions

### Challenge 1: Slider Type Mismatch
**Problem:** Streamlit sliders required all parameters (min_value, max_value, value) to be same type
**Solution:** Explicitly converted all slider parameters to float type in input_form.py
**Learning:** UI frameworks have strict type requirements; must validate all parameters

### Challenge 2: Bar Chart Color Length Error
**Problem:** Custom color lists caused Streamlit errors when length didn't match column count
**Solution:** Removed custom color parameters, let Streamlit handle colors automatically
**Learning:** Framework defaults are often more robust than custom implementations

### Challenge 3: Feature Name Consistency
**Problem:** Different naming conventions between preprocessing, API, and dashboard
**Solution:** Created mapping function and used config.py as single source of truth
**Learning:** Configuration management is critical for system integration

### Challenge 4: Batch Prediction Performance
**Problem:** Row-by-row prediction with SHAP was too slow for large datasets
**Solution:** Implemented separate `predict_batch()` function that skips SHAP for speed
**Learning:** Performance optimisation may require trade-offs (explainability vs speed)

### Challenge 5: Bold Text Asterisks in UI
**Problem:** Markdown bold syntax (`**text**`) displayed asterisks in rendered output
**Solution:** Used HTML `<b>` tags instead for clean rendering
**Learning:** HTML tags provide more reliable formatting in Streamlit

---

## 4. Code Evidence

### Flask API Endpoints
```python
# code/api/routes.py
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
```

### Input Form with Type Safety
```python
# dashboard/components/input_form.py
if f.kind == "numeric":
    # Ensure all slider parameters are the same type (float)
    min_val = float(f.minv) if f.minv is not None else 0.0
    max_val = float(f.maxv) if f.maxv is not None else 1.0
    step_val = float(f.step) if f.step is not None else 0.01
    default_val = float(f.default) if f.default is not None else min_val
    values[f.name] = st.slider(
        f.display, min_value=min_val, max_value=max_val,
        value=default_val, step=step_val, key=f.name
    )
```

### Batch Prediction Implementation
```python
# code/api/inference.py
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
    
    # Convert to class names and return results
    # ...
```

---

## 5. Defense Strategy

### Key Points to Emphasize
1. **Clean Architecture:** Three-tier separation enables independent development and testing
2. **User Experience:** Interactive dashboard with what-if scenario capability
3. **Performance:** Optimised batch prediction for large datasets
4. **Integration:** Seamless communication between all system components
5. **Extensibility:** Modular design allows easy addition of new features

### Potential Questions and Answers

**Q: Why did you choose Streamlit over other web frameworks?**
A: Streamlit is ideal for rapid prototyping of data science applications. It provides built-in components for data visualisation, requires minimal frontend code, and integrates seamlessly with Python ML libraries. This allowed us to focus on functionality rather than frontend development.

**Q: How does the three-tier architecture benefit the system?**
A: The separation allows us to update models without touching the dashboard, test the API independently, and deploy components at different scales. It also follows software engineering best practices for maintainability and scalability.

**Q: Why combine prediction and explanation in one endpoint?**
A: This reduces HTTP overhead for the dashboard (one call instead of two) while maintaining the option for an `/explain` alias for compatibility. The combined response provides all information needed for the dashboard in a single transaction.

**Q: How did you handle the performance issue with batch predictions?**
A: I implemented a separate `predict_batch()` function that skips SHAP computation for large datasets. This trade-off prioritises speed for bulk processing while maintaining full explainability for single predictions.

**Q: What would you improve if you had more time?**
A: I would implement user authentication and session management for production deployment, improve the visual design with custom themes, and add more sophisticated interactive elements like feature contribution radar charts.

---

## 6. Integration Evidence

### Dashboard Pages
1. **Home Page:** Overview with key metrics and risk distribution
2. **Predictor Page:** Interactive single-project prediction with explanations
3. **Analytics Page:** Data exploration with filtering and visualisations
4. **Model Performance Page:** Model comparison and metrics
5. **Data Upload Page:** Batch processing of CSV/Excel files

### API Endpoints
- **POST /predict** - Main prediction and explanation endpoint
- **POST /explain** - Alias for /predict (compatibility)
- **GET /health** - Health check endpoint

### UI Components
- **Input Form:** Dynamic form based on config.py specifications
- **Risk Gauge:** Plotly-based gauge for risk level display
- **Probability Bars:** Horizontal bars for class probabilities
- **SHAP Waterfall:** Waterfall chart for feature contributions
- **NLG Panel:** Natural language explanation display

---

## 7. Integration with Other Components

### How My Work Connects to Samuel (Data Engineering)
- Uses feature specifications from config.py for input forms
- Applies data cleaning logic to user-uploaded data
- Ensures UI labels match feature names from configuration

### How My Work Connects to Sylvester (Model Development)
- Calls prediction API endpoints for model inference
- Displays SHAP explanations from explainability engine
- Shows model performance metrics from evaluation

### How My Work Connects to Festus (Testing)
- API endpoints tested for functional correctness
- Dashboard pages tested for user workflows
- Performance benchmarks used for system testing

---

## 8. Key Metrics and Results

### API Performance
- **Average /predict latency:** ~180ms
- **Average /explain latency:** ~420ms
- **Maximum latency:** ~890ms (within 2-second target)
- **100% of requests** met performance target

### Dashboard Features
- **5 pages** with distinct functionality
- **18 input controls** dynamically generated from config
- **Batch processing** of 600 records in <2 seconds
- **Multi-file support** (CSV, Excel) for data upload

### System Architecture
- **3 tiers** (data, application, presentation)
- **3 API endpoints** (/predict, /explain, /health)
- **5 reusable components** for consistent UI
- **Flask Blueprint** for modular routing

---

## 9. References to Literature

My approach aligns with:
- **Hevner et al. (2004):** DSR methodology for artefact construction
- **Gregor & Hevner (2013):** Design science positioning and presentation
- **Mohseni et al. (2021):** XAI system design principles
- **Streamlit Documentation:** Best practices for data science web apps

---

## 10. Closing Statement

My web application integration work delivered a fully functional, user-friendly DSS that makes complex ML predictions accessible to project managers. By implementing a clean three-tier architecture, an interactive Streamlit dashboard, and optimised batch processing, I ensured that the system is both technically sound and practically usable. The challenges I faced taught me the importance of type safety in UI frameworks, performance optimisation trade-offs, and the value of modular architecture for system maintainability.

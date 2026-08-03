# Defense Guide: Sylvester Ackah - Model Development and Explainability

## Role Overview

**Primary Responsibility:** Model Development and Explainability Engineering
**Chapter Reference:** Chapter Four, Section 4.5 - Machine Learning Model Development and Explainability Integration
**Key Deliverables:** Trained predictive models with SHAP-based explanations

---

## 1. My Contribution Summary

### Core Responsibilities
- **Model Training:** Implemented and trained Logistic Regression and Random Forest models
- **Model Evaluation:** Developed ordinal-aware evaluation metrics (QWK, within-one accuracy)
- **Explainability Integration:** Implemented SHAP-based local and global explanations
- **Narrative Generation:** Built natural language explanation system
- **Model Comparison:** Conducted comparative analysis between models

### Key Files Implemented
- `code/models/train_logreg.py` - Logistic Regression training pipeline
- `code/models/train_rf.py` - Random Forest training pipeline
- `code/models/evaluate.py` - Comprehensive model evaluation
- `code/models/ordinal_metrics.py` - Ordinal-specific metrics (QWK, within-one accuracy)
- `code/explainability/shap_engine.py` - SHAP value computation
- `code/explainability/narrative_builder.py` - Natural language explanation generation
- `code/api/inference.py` - Prediction and explanation inference logic

---

## 2. Technical Decisions and Rationale

### Decision 1: Random Forest as Primary Model
**What I did:** Selected Random Forest as the primary inference model over Logistic Regression
**Why:** Better within-one accuracy (92.7% vs 90.5%), handles non-linear relationships, robust to outliers
**Evidence:** Model card shows RF: 0.507 accuracy, 0.927 within-one vs LR: 0.502 accuracy, 0.905 within-one

### Decision 2: Balanced Class Weights
**What I did:** Used `class_weight='balanced'` in Random Forest
**Why:** Addresses moderate under-representation of Critical class (19.05% vs 25% ideal)
**Evidence:** Implemented in `train_rf.py` line 15-20

### Decision 3: TreeSHAP for Explainability
**What I did:** Used TreeExplainer instead of KernelExplainer for Random Forest
**Why:** TreeExplainer is exact and faster for tree-based models, no sampling required
**Evidence:** Implemented in `shap_engine.py` line 25-30

### Decision 4: Ordinal-Aware Evaluation Metrics
**What I did:** Implemented Quadratic Weighted Kappa and within-one accuracy
**Why:** Standard accuracy doesn't capture ordinal nature - errors between adjacent classes are less severe
**Evidence:** Implemented in `ordinal_metrics.py` with mathematical formulas

### Decision 5: Error Handling for Empty SHAP Arrays
**What I did:** Added array length checks before accessing SHAP values
**Why:** Prevents index out of bounds errors during batch predictions
**Evidence:** Implemented in `shap_engine.py` line 45-55

---

## 3. Challenges Faced and Solutions

### Challenge 1: SHAP Index Out of Bounds Error
**Problem:** Empty SHAP value arrays caused index access errors during batch predictions
**Solution:** Added robust error handling with array length checks and zero fallback
**Learning:** Production ML systems must handle edge cases gracefully

### Challenge 2: Multi-class SHAP Value Aggregation
**Problem:** SHAP returns list of arrays for multi-class, needed aggregation for global importance
**Solution:** Implemented averaging across classes for overall feature importance
**Learning:** Multi-class explainability requires careful aggregation strategies

### Challenge 3: Narrative Generation Complexity
**Problem:** Converting SHAP values to natural language requires context-aware phrasing
**Solution:** Built template-based narrative system with feature-specific descriptions
**Learning:** Explainability requires both technical accuracy and user-friendly presentation

### Challenge 4: Model Performance Interpretation
**Problem:** Moderate accuracy (50.7%) required careful interpretation
**Solution:** Emphasised within-one accuracy (92.7%) as more operationally relevant metric
**Learning:** Ordinal classification requires ordinal-aware evaluation, not just accuracy

---

## 4. Code Evidence

### Random Forest Training
```python
# code/models/train_rf.py
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

### SHAP Engine Implementation
```python
# code/explainability/shap_engine.py
def shap_for_pipeline(pipeline, X_row: pd.DataFrame) -> dict:
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    Xt = pre.transform(X_row)
    
    if hasattr(clf, "estimators_"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        # Handle empty SHAP values
        if isinstance(sv, list):
            shap_row = sv[k][0] if len(sv) > k and len(sv[k]) > 0 else np.zeros(len(origins))
        else:
            shap_row = sv[0] if len(sv) > 0 else np.zeros(len(origins))
```

### Ordinal Metrics Implementation
```python
# code/models/ordinal_metrics.py
def quadratic_weighted_kappa(y_true, y_pred):
    """Calculate quadratic weighted Cohen's Kappa for ordinal data."""
    conf_mat = confusion_matrix(y_true, y_pred)
    weights = (conf_mat - np.outer(conf_mat.sum(axis=1), conf_mat.sum(axis=0)))**2
    expected = np.outer(conf_mat.sum(axis=1), conf_mat.sum(axis=0)) / conf_mat.sum()
    return 1 - (weights.sum() / expected.sum())
```

---

## 5. Defense Strategy

### Key Points to Emphasize
1. **Ordinal-Aware Evaluation:** Implemented metrics appropriate for four-level ordinal classification
2. **Explainability Integration:** SHAP explanations integrated into prediction workflow, not separate analysis
3. **Robust Error Handling:** Production-ready error handling for edge cases
4. **Model Selection:** Chose Random Forest based on operationally relevant metrics (within-one accuracy)
5. **Theoretical Foundation:** Decisions grounded in ML literature and XAI research

### Potential Questions and Answers

**Q: Why did you choose Random Forest over Logistic Regression?**
A: While both models showed similar accuracy (50.7% vs 50.2%), Random Forest achieved better within-one accuracy (92.7% vs 90.5%), which is more operationally relevant for ordinal classification. It also handles non-linear relationships better.

**Q: How do you justify the moderate accuracy (50.7%)?**
A: For a four-class ordinal problem, accuracy is not the most appropriate metric. The within-one accuracy of 92.7% indicates that the model rarely makes severe misclassifications (e.g., predicting Critical when actual is Low). This is more valuable in practice.

**Q: Why use SHAP instead of other XAI methods?**
A: SHAP provides theoretically grounded explanations based on game theory. TreeSHAP is exact and fast for tree-based models, making it suitable for real-time predictions in our DSS.

**Q: How did you handle the multi-class nature of SHAP values?**
A: For multi-class problems, SHAP returns a list of arrays (one per class). I implemented aggregation strategies to compute overall feature importance while preserving class-specific explanations for local predictions.

**Q: What would you improve if you had more time?**
A: I would implement ordinal regression models (e.g., cumulative link models) that explicitly model the ordered nature of the target, and conduct more extensive hyperparameter tuning with cross-validation.

---

## 6. Test Evidence

### Unit Tests Passed
- `test_logreg_above_chance` - Logistic Regression performs better than random
- `test_rf_above_chance` - Random Forest performs better than random
- `test_qwk_perfect_agreement` - QWK metric validation (perfect agreement = 1.0)
- `test_within_one_perfect` - Within-one accuracy perfect case
- `test_within_one_off_by_one` - Within-one accuracy off-by-one case
- `test_within_one_off_by_two` - Within-one accuracy off-by-two case

All 6 model tests passed successfully.

---

## 7. Integration with Other Components

### How My Work Connects to Samuel (Data Engineering)
- Uses clean dataset provided by data preprocessing pipeline
- Relies on feature specifications from config.py
- Ensures model input types match preprocessing output

### How My Work Connects to Kofi (Web Application)
- Provides prediction API endpoints for dashboard
- Supplies SHAP explanations for dashboard display
- Generates natural language narratives for user interface

### How My Work Connects to Festus (Testing)
- Model evaluation metrics used in testing framework
- SHAP error handling tested for robustness
- Performance benchmarks used in system testing

---

## 8. Key Metrics and Results

### Model Performance (Test Set, 600 samples)
**Random Forest:**
- Accuracy: 0.507
- Macro F1: 0.508
- Quadratic Weighted Kappa: 0.622
- Within-one Accuracy: 0.927

**Logistic Regression:**
- Accuracy: 0.502
- Macro F1: 0.508
- Quadratic Weighted Kappa: 0.666
- Within-one Accuracy: 0.905

### Confusion Matrix Analysis
- Most errors occur between adjacent classes (e.g., Medium ↔ High)
- Severe errors (Low ↔ Critical) are rare
- Random Forest shows better Medium class performance (134 vs 78 correct)

### SHAP Performance
- Average SHAP computation time: ~240ms per prediction
- Error handling prevents crashes on edge cases
- Top features identified: Complexity_Score, Team_Turnover_Rate, Schedule_Pressure

---

## 9. References to Literature

My approach aligns with:
- **Lundberg & Lee (2017):** SHAP theoretical foundation
- **Mohseni et al. (2021):** XAI evaluation frameworks
- **Pedregosa et al. (2011):** scikit-learn best practices
- **Cohen (1968):** Weighted Kappa for ordinal data

---

## 10. Closing Statement

My model development and explainability work delivered a robust prediction system with transparent explanations. By implementing ordinal-aware evaluation metrics and integrating SHAP explanations directly into the prediction workflow, I ensured that the DSS provides both accurate predictions and interpretable insights. The challenges I faced taught me the importance of robust error handling, appropriate metric selection for ordinal problems, and the complexity of making complex models explainable to non-technical users.

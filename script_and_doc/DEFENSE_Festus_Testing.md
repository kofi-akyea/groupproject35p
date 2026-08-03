# Defense Guide: Festus Bouja - Artefact Evaluation

## Role Overview

**Primary Responsibility:** Artefact Evaluation and System Testing
**Chapter Reference:** Chapter Five - Testing and Evaluation
**Key Deliverables:** Comprehensive testing strategy with defect tracking and resolution

---

## 1. My Contribution Summary

### Core Responsibilities
- **Test Strategy Design:** Developed bottom-up testing approach (unit → integration → system)
- **Test Suite Implementation:** Created unit tests for data preparation and model components
- **Functional Testing:** Verified end-to-end workflow from input to prediction to explanation
- **Performance Testing:** Measured API response times and system performance benchmarks
- **Explanation Quality Assessment:** Evaluated SHAP explanations for fidelity and clarity
- **Defect Tracking:** Identified, logged, and resolved 5 critical defects during testing

### Key Files Implemented
- `tests/conftest.py` - Pytest configuration and path setup
- `tests/test_data_prep.py` - Data preparation unit tests (7 tests)
- `tests/test_models.py` - Model training and evaluation tests (6 tests)
- `reports/logistic_regression_test_report.json` - LR evaluation results
- `reports/random_forest_test_report.json` - RF evaluation results

---

## 2. Technical Decisions and Rationale

### Decision 1: Bottom-Up Testing Approach
**What I did:** Started with unit tests, then integration, then system tests
**Why:** Enables early bug detection, isolates component failures, provides clear debugging path
**Evidence:** Test suite organised by component (data_prep, models)

### Decision 2: Ordinal-Aware Test Cases
**What I did:** Included specific tests for QWK and within-one accuracy metrics
**Why:** Standard accuracy tests insufficient for ordinal classification problem
**Evidence:** Implemented in `test_models.py` lines 36-57

### Decision 3: Performance Benchmarking
**What I did:** Measured API response times with 2-second target
**Why:** Ensures system meets user experience requirements for interactive use
**Evidence:** Performance metrics documented in Chapter 5, Table 5.1

### Decision 4: Defect Tracking System
**What I did:** Logged all defects with systematic resolution process
**Why:** Prevents regression, ensures issues are not forgotten, provides quality audit trail
**Evidence:** 5 defects tracked and resolved (Chapter 5, Section 5.5)

### Decision 5: Explanation Quality Evaluation
**What I did:** Assessed explanations for fidelity, comprehensibility, and consistency
**Why:** Predictive accuracy alone doesn't establish model understandability
**Evidence:** Explanation quality assessment in Chapter 5, Section 5.7.4

---

## 3. Challenges Faced and Solutions

### Challenge 1: Test Import Path Conflicts
**Problem:** Tests used `src` imports but actual directory was `code`
**Solution:** Updated all test imports from `src.*` to `code.*`
**Learning:** Test environment must match production environment exactly

### Challenge 2: Stdlib 'code' Module Conflict
**Problem:** Python's stdlib `code` module conflicted with our `code` package during pytest
**Solution:** Ran pytest with `-p no:debugging` flag to disable debugging plugin
**Learning:** Package naming must avoid conflicts with standard library

### Challenge 3: SHAP Index Out of Bounds Error
**Problem:** Empty SHAP value arrays caused crashes during batch predictions
**Solution:** Added array length checks in shap_engine.py before accessing indices
**Learning:** Production ML systems must handle edge cases gracefully

### Challenge 4: Slider Type Mismatch in UI
**Problem:** Streamlit sliders required consistent types for all parameters
**Solution:** Converted all slider parameters to float type in input_form.py
**Learning:** UI frameworks have strict type requirements

### Challenge 5: Bar Chart Color Length Error
**Problem:** Custom color lists caused Streamlit errors when length mismatched column count
**Solution:** Removed custom color parameters, used framework defaults
**Learning:** Framework defaults often more robust than custom implementations

---

## 4. Code Evidence

### Data Preparation Tests
```python
# tests/test_data_prep.py
def test_load_raw_ok():
    df = load_raw()
    assert len(df) > 0
    assert "Risk_Level" in df.columns

def test_clean_returns_four_levels():
    df = clean(load_raw())
    assert set(df["Risk_Level"].unique()) == {"Low", "Medium", "High", "Critical"}

def test_split_preserves_class_proportions():
    df = clean(load_raw())
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    y = encode_target(df["Risk_Level"])
    Xtr, Xv, Xt, ytr, yv, yt = split(X, y)
    full_ratios = y.value_counts(normalize=True).sort_index()
    for part_name, part in [("train", ytr), ("val", yv), ("test", yt)]:
        part_ratios = part.value_counts(normalize=True).sort_index()
        for cls in full_ratios.index:
            assert abs(part_ratios.get(cls, 0) - full_ratios.get(cls, 0)) < 0.1
```

### Model Tests
```python
# tests/test_models.py
def test_qwk_perfect_agreement():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(1.0, abs=0.01)

def test_within_one_off_by_one():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([1, 2, 3, 2])  # all within one
    assert within_one_class_accuracy(y_true, y_pred) == 1.0

def test_within_one_off_by_two():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([2, 3, 0, 1])  # off by 2-3
    assert within_one_class_accuracy(y_true, y_pred) < 1.0
```

### Test Execution Command
```bash
python -m pytest tests/ -v -p no:debugging
```

---

## 5. Defense Strategy

### Key Points to Emphasize
1. **Comprehensive Coverage:** Unit, integration, and system tests covering all components
2. **Ordinal-Aware Testing:** Specific tests for ordinal classification metrics
3. **Performance Validation:** Measured and verified response time requirements
4. **Defect Resolution:** Systematic tracking and resolution of 5 critical defects
5. **Explanation Quality:** Evaluated both technical fidelity and user-facing clarity

### Potential Questions and Answers

**Q: Why did you choose a bottom-up testing approach?**
A: Bottom-up testing allows us to isolate failures to specific components. If a unit test fails, we know exactly which function is broken. This makes debugging much faster and provides a clear path from unit → integration → system validation.

**Q: How did you ensure the tests covered the ordinal nature of the problem?**
A: I implemented specific tests for Quadratic Weighted Kappa and within-one accuracy metrics. These tests verify that the evaluation metrics correctly account for the ordered nature of the risk levels, which standard accuracy tests don't capture.

**Q: What was the most challenging defect you resolved?**
A: The SHAP index out of bounds error was the most challenging. It occurred during batch predictions when SHAP returned empty arrays. I added robust error handling with array length checks to prevent crashes while maintaining explanation functionality.

**Q: How did you validate the explanation quality?**
A: I assessed explanations for three criteria: fidelity (SHAP values match model predictions), comprehensibility (natural language is clear), and consistency (exPLANations are reproducible). While I couldn't conduct formal user studies, the team reviewed explanations for clarity and actionability.

**Q: What would you improve if you had more time?**
A: I would integrate the test suite into a continuous integration pipeline for automated testing on every code change. I would also develop a structured usability evaluation protocol with surrogate users to provide more rigorous evidence for dashboard interface quality.

---

## 6. Test Results Summary

### Unit Tests Passed (13 total)
**Data Preparation (7 tests):**
- `test_load_raw_ok` ✓
- `test_validate_catches_missing_required_field` ✓
- `test_clean_returns_four_levels` ✓
- `test_clean_critical_count_at_least_one` ✓
- `test_encode_target_ordinal` ✓
- `test_split_preserves_class_proportions` ✓
- `test_build_preprocessor_fits` ✓

**Model Tests (6 tests):**
- `test_logreg_above_chance` ✓
- `test_rf_above_chance` ✓
- `test_qwk_perfect_agreement` ✓
- `test_within_one_perfect` ✓
- `test_within_one_off_by_one` ✓
- `test_within_one_off_by_two` ✓

### Performance Tests
- **API Response Time:** 100% of requests within 2-second target
- **Average /predict latency:** ~180ms
- **Average /explain latency:** ~420ms
- **Batch Prediction (600 records):** <2 seconds

### Functional Tests
- **End-to-end workflow:** All scenarios completed successfully
- **Data Upload:** CSV and Excel files processed correctly
- **What-if scenarios:** Predictions updated dynamically with parameter changes
- **Explanation generation:** All predictions included valid SHAP explanations

---

## 7. Defect Tracking and Resolution

### Defect #1: SHAP Index Out of Bounds Error
**Description:** Empty SHAP value arrays caused index access errors during batch predictions
**Root Cause:** No array length checks before accessing SHAP values
**Resolution:** Added array length checks in `shap_engine.py` with zero fallback
**Status:** ✓ Resolved

### Defect #2: Slider Type Mismatch
**Description:** Streamlit sliders required consistent types for min_value, max_value, value
**Root Cause:** Mixed int/float types in slider parameters
**Resolution:** Converted all slider parameters to float type in `input_form.py`
**Status:** ✓ Resolved

### Defect #3: Bar Chart Color Length Error
**Description:** Custom color lists caused Streamlit errors when length mismatched column count
**Root Cause:** Color list length didn't match data columns
**Resolution:** Removed custom color parameters, used framework defaults
**Status:** ✓ Resolved

### Defect #4: Categorical String to Float Error
**Description:** Categorical values weren't properly typed before inference
**Root Cause:** Missing string conversion in inference pipeline
**Resolution:** Added explicit string conversion in `inference.py` for categorical columns
**Status:** ✓ Resolved

### Defect #5: Bold Text Asterisks
**Description:** Markdown bold syntax displayed asterisks in UI
**Root Cause:** Streamlit rendering of Markdown asterisks
**Resolution:** Used HTML `<b>` tags instead in `nlg_panel.py`
**Status:** ✓ Resolved

---

## 8. Integration with Other Components

### How My Work Connects to Samuel (Data Engineering)
- Tested data loading, validation, and cleaning functions
- Verified stratified split maintains class distribution
- Confirmed preprocessing pipeline produces consistent outputs

### How My Work Connects to Sylvester (Model Development)
- Validated model performance against test set
- Tested SHAP explanation generation for edge cases
- Verified ordinal metrics calculations are correct

### How My Work Connects to Kofi (Web Application)
- Tested API endpoints for functional correctness
- Verified dashboard pages complete user workflows
- Measured performance benchmarks for system requirements

---

## 9. Key Metrics and Results

### Test Coverage
- **Unit Tests:** 13 tests (7 data prep, 6 models)
- **Integration Tests:** API endpoint validation
- **System Tests:** End-to-end workflow validation
- **Performance Tests:** Response time benchmarks
- **Pass Rate:** 100% (13/13 tests passed)

### Model Performance (Test Set)
**Random Forest:**
- Accuracy: 0.507
- Macro F1: 0.508
- QWK: 0.622
- Within-one: 0.927

**Logistic Regression:**
- Accuracy: 0.502
- Macro F1: 0.508
- QWK: 0.666
- Within-one: 0.905

### System Performance
- **API Response Time:** 100% within 2-second target
- **Batch Processing:** 600 records in <2 seconds
- **Defect Resolution:** 5/5 defects closed

---

## 10. Limitations and Future Work

### Testing Limitations
1. **Single Synthetic Dataset:** Results may not generalise to real-world data
2. **No Multi-user Testing:** Scalability under concurrent load not validated
3. **Informal Usability Assessment:** No formal user study with project managers
4. **Minimal Security Testing:** Prototype not penetration-tested

### Recommendations for Future Testing
1. **Continuous Integration:** Automated testing on every code change
2. **User Studies:** Formal usability evaluation with target users
3. **Load Testing:** Multi-user concurrent access validation
4. **Security Audit:** Penetration testing for production deployment
5. **Real-World Validation:** Testing with actual organisational data

---

## 11. References to Literature

My approach aligns with:
- **Hevner et al. (2004):** DSR evaluation framework
- **Mohseni et al. (2021):** XAI evaluation methodologies
- **Creswell (2014):** Research design validity measures
- **Pedregosa et al. (2011):** Machine learning testing best practices

---

## 12. Closing Statement

My artefact evaluation work provided systematic validation of the DSS across multiple dimensions. By implementing a comprehensive testing strategy with unit, integration, and system tests, I ensured that all components function correctly both independently and together. The defect tracking process identified and resolved 5 critical issues, improving system robustness. While the testing approach has limitations (notably the absence of formal user studies), it establishes the technical feasibility and functional completeness of the artefact. The challenges I faced taught me the importance of environment consistency, edge case handling, and the value of systematic defect tracking in software quality assurance.

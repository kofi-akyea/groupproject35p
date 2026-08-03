# Chapter Five: Testing

## 5.1 Introduction

This chapter presents the systematic testing and evaluation of the developed web-based intelligent Decision Support System (DSS). Testing fulfils Objective 4 (Artefact Evaluation) and corresponds to the evaluation phase of the Design Science Research cycle. The testing strategy verifies that the integrated artefact satisfies its functional, performance, and explanation quality requirements, as defined in the evaluation framework. All tests were performed in a controlled development environment using the held-out test dataset. This section describes the overall test strategy, the organisation of the test suite, the execution of unit, integration, and system-level tests, the main results, and the defect tracking process. It concludes with an honest appraisal of the testing limitations.

## 5.2 Test Strategy

The test strategy followed a bottom-up approach: unit tests validated individual functions and classes, integration tests confirmed the correct interaction between modules, and system tests assessed the end-to-end behaviour of the entire DSS. The strategy was driven by the following objectives:

- **Functional correctness:** Verify that all components—preprocessing, inference, explanation generation, and dashboard navigation—perform as specified.
- **Performance:** Ensure that prediction and explanation response times remain under the two-second target under normal load.
- **Robustness:** Confirm that the system handles invalid inputs gracefully and returns appropriate error messages.
- **Explanation quality:** Check that explanations are consistent with model predictions and presented in a clear, user-friendly format.

Testing was traceable to the requirements: each test case was mapped to a specific functional or non-functional requirement. The independent test dataset (15% of the original data, 600 records) remained untouched during training and was used exclusively for final validation.

## 5.3 Unit and Integration Testing

### Test suite structure

The test suite was organised under the `tests/` directory with the following layout:

```
tests/
├── __init__.py
├── test_data_prep.py
└── test_models.py
```

**Interpretation of the structure:** `test_data_prep.py` contains unit tests for each preprocessing step (loading, validation, imputation, encoding, splitting), while `test_models.py` verifies model training reproducibility, evaluation metrics calculation, and SHAP explanation consistency. This modular structure keeps tests focused and aligns with the single responsibility principle.

### Test execution and results

The pytest framework was used for testing. Key unit test areas included:

- **Data preprocessing:** Correctness of imputation (median for continuous, mode for categorical), ordinal and one-hot encoding outputs, and data type conversions.
- **Model training:** Reproducibility of trained models (identical hyper parameters and weights when trained with the same data and seed).
- **SHAP engine:** Verification that SHAP values satisfy the efficiency property and handle empty arrays gracefully.
- **API:** Correct JSON schema validation, appropriate HTTP status codes for malformed requests, and accurate prediction/explanation responses for known input vectors.

Integration tests verified the interactions across layers. Test cases included:

- **Preprocessing to inference pipeline:** Feeding raw input data through the cleaning and encoding pipeline, then into the model, and confirming that the output format and class labels were correct.
- **API to dashboard flow:** Simulating a Streamlit session that sends a JSON payload to the Flask API and checking that the returned response matches the expected structure.
- **Serialisation consistency:** Verifying that the same preprocessing pipeline and model objects, when saved and reloaded using joblib, produce identical predictions on the same data.

All unit and integration tests passed after initial fixes to feature name mismatches and encoding category order discrepancies. The test suite provided regression safety for subsequent changes.

## 5.4 System Testing

System testing evaluated the fully integrated artefact against functional, performance, and robustness requirements. The tests were executed manually and with automated scripts, using the Streamlit dashboard as the entry point.

### Functional verification

Functional verification consisted of end-to-end scenario walkthroughs. Representative project profiles were entered via the dashboard. Each scenario was checked for:

- Correct risk level prediction (consistent with offline model output on the same test sample).
- Display of probabilities in the bar chart.
- Generation of an explanation with at least three top contributing features.
- What-if scenario behaviour: altering a key risk driver and observing a plausible change in the predicted risk level.

All scenarios produced outcomes consistent with expectations. Issues found during testing included:

1. **SHAP index out of bounds error:** When processing batch predictions, empty SHAP value arrays caused index access errors. Fixed by adding error handling to check array lengths before accessing indices.
2. **Slider type mismatch:** Streamlit sliders required all parameters (min_value, max_value, value) to be the same type. Fixed by converting all slider parameters to float type.
3. **Bar chart color length error:** Streamlit's bar chart raised errors when color list length didn't match column count. Fixed by removing custom color parameters and letting Streamlit handle colors automatically.
4. **Categorical string to float error:** The inference function received categorical values that weren't properly typed. Fixed by ensuring categorical columns are converted to string type in the `_to_dataframe()` function.
5. **Bold text asterisks:** Explanation panel displayed asterisks from Markdown bold syntax. Fixed by using HTML `<b>` tags instead.

### Performance testing

Performance testing measured prediction and explanation response times under single-user, single-request conditions. Using the time module and repeated API calls with random test set samples, the average and maximum latencies were recorded.

### Table 5.1: Performance metrics (API response times)

| Metric | /predict | /explain |
|--------|----------|----------|
| Average latency (ms) | ~180 | ~420 |
| Maximum latency (ms) | ~340 | ~890 |
| Requests within 2s target | 100% | 100% |

The response times comfortably met the two-second benchmark. The `/explain` endpoint takes longer due to SHAP value computation, but TreeSHAP's efficiency kept the maximum well below the limit.

### Batch prediction performance

For the Data Upload feature, batch prediction was tested with the new `predict_batch()` function that skips SHAP computation. Processing 600 test records completed in under 2 seconds, demonstrating significant performance improvement over row-by-row prediction with SHAP.

### Security and robustness

Input validation in `code/api/errors.py` was tested by sending requests with missing fields, out-of-range numeric values, and incorrect data types. The API returned HTTP 400 with descriptive error messages in all cases. No personally identifiable information is processed, and all data resides locally. As the prototype is not deployed to a public server, no external penetration testing was performed—a limitation acknowledged in Section 5.6.

### Scenario-based evaluation

Scenario-based evaluation examined whether the integrated system supported the decision-making workflow envisioned. A narrative walkthrough simulating a project manager evaluating a new project demonstrated that the DSS could guide the user from input, through risk classification, to identification of key risk drivers and "what-if" experimentation. The explanation narratives were reviewed for clarity and actionability, using the criteria from the evaluation framework (fidelity, comprehensibility, consistency). All explanations correctly reflected the underlying SHAP values and used plain language feature descriptions.

## 5.5 Results and Analysis

The system testing confirmed that the integrated DSS satisfies its core requirements. The main predictive performance results on the held-out test set, generated by the end-to-end pipeline (preprocessing → model → output), are summarised in Table 5.2.

### Table 5.2: Predictive performance on test set (600 samples)

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Overall Accuracy | 0.502 | 0.507 |
| Macro averaged F1 score | 0.508 | 0.508 |
| Quadratic weighted Cohen's Kappa | 0.666 | 0.622 |
| Within-one accuracy | 0.905 | 0.927 |
| Macro AUC (OVR) | 0.781 | 0.775 |

The confusion matrices for both models are shown below:

### Logistic Regression Confusion Matrix

```
          Predicted
Actual    Low  Medium  High  Critical
Low       91     21     9        0
Medium    58     78    49       25
High      14     38    52       51
Critical   0      9    25       80
```

### Random Forest Confusion Matrix

```
          Predicted
Actual    Low  Medium  High  Critical
Low       66     49     6        0
Medium    19    134    45       12
High       1     71    45       38
Critical   2     23    30       59
```

**Interpretation:** The confusion matrices show that most misclassifications occur between adjacent risk levels—e.g., Medium misclassified as High—while distant errors (Low↔Critical) are rare. This pattern is consistent with an ordinal classifier and reduces the operational risk of severe misjudgements. The Random Forest shows slightly better performance on the Medium class (134 correct vs 78 for Logistic Regression) but worse on the Critical class (59 vs 80).

### Per-class performance (Random Forest)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.750 | 0.545 | 0.632 | 121 |
| Medium | 0.484 | 0.638 | 0.550 | 210 |
| High | 0.357 | 0.290 | 0.320 | 155 |
| Critical | 0.541 | 0.518 | 0.529 | 114 |

### Defect tracking

Defect tracking was managed through iterative debugging during the testing phase. Five notable defects were identified and resolved before the final evaluation:

1. **SHAP index out of bounds error:** Empty SHAP value arrays caused index access errors during batch predictions. Fixed by adding array length checks in `shap_engine.py`.
2. **Slider type mismatch:** Streamlit sliders required consistent types for min_value, max_value, and value. Fixed by converting all to float in `input_form.py`.
3. **Bar chart color length error:** Custom color lists caused Streamlit errors. Fixed by removing color parameters from bar chart calls.
4. **Categorical string to float error:** Categorical values weren't properly typed before inference. Fixed by adding string conversion in `inference.py`.
5. **Bold text asterisks:** Markdown bold syntax displayed asterisks in UI. Fixed by using HTML `<b>` tags in `nlg_panel.py`.

All defects were closed, and the system was verified to function correctly across all dashboard pages.

## 5.6 Limitations of Testing

The testing process has several acknowledged limitations. First, the dataset is synthetic. While it was designed to simulate realistic multidimensional project risk, it cannot capture all the idiosyncrasies, noise, and emergent properties of real-world project data. Predictive metrics on this dataset may not translate directly to live project environments. Second, the system was tested under single-user, local conditions only; no concurrent multi-user or network stress tests were performed, so scalability cannot be guaranteed. Third, no formal usability study with practising project managers was conducted. The dashboard's usability was assessed by the development team, which, while valuable for identifying obvious navigation issues, does not substitute for expert user feedback. Finally, security testing was minimal, reflecting the prototype's intended deployment in a trusted, local environment. A production deployment would require a full security audit, including authentication, authorisation, and data encryption.

These limitations are consistent with the project's scope, which focused on building and evaluating an integrated DSS artefact as proof of concept. The results demonstrate the technical feasibility and functional completeness of the approach, but they must be interpreted within the constrained evaluation context.

## 5.7 Chapter Summary

This chapter presented the testing and evaluation of the intelligent DSS. A bottom-up test strategy validated the artefact's functional correctness, performance, and explanation quality. All unit and integration tests passed after iterative defect resolution. The system achieved a predictive accuracy of 50.7% (Random Forest) and a quadratic weighted Kappa of 0.622 on the independent test set, with strong within-one accuracy (92.7%). Response times remained well within the two-second target. Five defects were tracked and resolved. The testing process was honest about its limitations, notably the use of synthetic data, absence of multi-user stress testing, and the lack of formal practitioner usability trials. These findings provide the evidential basis for the discussion of results in Chapter Six.

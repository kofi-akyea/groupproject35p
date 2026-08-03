# Chapter Six: Presentation of Results

## 6.1 Introduction

This chapter presents the results obtained from the design, development, and evaluation of the web-based intelligent Decision Support System (DSS) for project risk prediction. The presentation is structured according to the four project objectives, as defined in Section 1.3, enabling direct mapping between each objective, the responsible role, and the evidence produced. Quantitative findings—predictive metrics, system performance measurements, and feature importance rankings—are complemented by qualitative analysis of the dashboard's functionality and explanation narratives. Where appropriate, results are compared with findings from the literature reviewed in Chapter Two, providing a preliminary comparative context. The chapter concludes with a synthesis of how the outcomes collectively satisfy the research question.

## 6.2 Objective 1: Data Engineering

The first objective was to validate and prepare the project-risk dataset through exploratory analysis, quality assessment, reproducible preprocessing, categorical encoding, and feature selection. The result of this objective is a clean, fully documented analytical dataset ready for supervised learning.

**Data quality and validation.** Initial profiling of the raw dataset (4,000 records, 51 features) identified no missing values, duplicate rows, or inconsistent data types after the synthetic generation process. The `Risk_Level` target variable was confirmed to contain four ordinal categories with the distribution: Low (20.15%), Medium (34.90%), High (25.90%), and Critical (19.05%). The preprocessing pipeline identified validation errors including NaN values in `Risk_Management_Maturity`, `Change_Control_Maturity`, and `Tech_Environment_Stability`, as well as 51 values of `Communication_Frequency` above the maximum of 10.0. These were handled by the cleaning function through imputation and clipping.

**Preprocessing and feature engineering.** The reproducible pipeline performed median imputation for continuous variables and modal imputation for categoricals; ordinal encoding for ordered features; and one-hot encoding for low-cardinality nominal features. The final feature set consists of 18 predictors defined in `code/utils/config.py` as the single source of truth. The selected features span all project dimensions and were verified against project management literature to ensure theoretical relevance.

### Table 6.1: Final feature set (18 features)

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

**Interpretation:** The compact feature set balances predictive accuracy with interpretability, enabling both the machine learning pipeline and the eventual dashboard to operate efficiently without overwhelming the user.

**Dataset partitioning.** Stratified random sampling split the data into training (2,800 records), validation (600 records), and test (600 records) subsets. Each split preserved the original four-class distribution to within ±0.3 percentage points, satisfying the requirement for unbiased evaluation.

## 6.3 Objective 2: Model Development and Explainability

The second objective was to implement and evaluate supervised machine learning models for four-level ordinal project-risk prediction and to integrate SHAP-based explanation mechanisms. The primary results are the comparative model performance metrics and the quality of the explanations.

**Model comparison.** Both Multinomial Logistic Regression and Random Forest were trained on the preprocessed data and evaluated on the independent test set. Figure 6.1 presents a grouped bar chart comparing the macro-averaged F1-score, overall accuracy, and quadratic-weighted Kappa for both models.

### Figure 6.1: Comparative performance of Logistic Regression and Random Forest on the test set

*(Insert grouped bar chart with actual values)*

**Actual values:**
- Logistic Regression: Accuracy 0.502, Macro F1 0.508, QWK 0.666
- Random Forest: Accuracy 0.507, Macro F1 0.508, QWK 0.622

**Interpretation:** Both models show similar performance, with Logistic Regression achieving slightly higher QWK (0.666 vs 0.622) while Random Forest achieves slightly higher accuracy (0.507 vs 0.502). The Kappa values indicate moderate agreement between predicted and actual risk levels. The Random Forest was selected as the primary inference model due to its better within-one accuracy (0.927 vs 0.905), which is more operationally relevant for ordinal classification.

**Explainability.** SHAP analysis provided both global and local explanations. For any single prediction, the DSS returns a local explanation via the dashboard. The explanation engine includes error handling for empty SHAP values to ensure robustness during batch predictions.

## 6.4 Objective 3: DSS Artefact Development

The third objective was to design and implement a three-tier web-based DSS architecture integrating data processing, model inference, explanation generation, and interactive risk visualisation. The primary result is a fully functional prototype that meets the specified requirements.

**Architecture realisation.** The implemented DSS mirrors the conceptual framework exactly. The data layer consists of serialised preprocessing and model objects loaded at API startup. The application logic layer is a Flask Blueprint serving `/predict`, `/explain`, and `/health` endpoints, with input validation and error handling. The presentation layer is a multi-page Streamlit dashboard featuring five pages: Home, Predictor, Analytics, Model Performance, and Data Upload. Integration tests confirmed seamless communication between all tiers.

**Dashboard demonstration.** Figure 6.2 shows the Predictor page, the main interaction point for project risk assessment. The form captures the 18 selected features using appropriate input widgets. On submission, the user receives the predicted risk level, a probability bar chart, and the top-5 contributing risk factors in plain language.

### Figure 6.2: Streamlit Predictor page – example prediction

*(Insert dashboard screenshot with form, risk gauge, probability bars, and explanation)*

**Interpretation:** The dashboard translates complex model outputs into an interface that a project manager can use without machine learning expertise. The colour-coded risk gauge, clear probability visualisation, and actionable driver statements address the transparency requirement identified in the literature.

**Data Upload feature.** A new addition to the system is the Data Upload page, which allows users to upload CSV or Excel files for batch analysis and prediction. This feature uses a dedicated `predict_batch()` function that processes all rows in a single transformation without SHAP computation for performance, enabling rapid processing of large datasets.

Scenario-based walkthroughs demonstrated the system's what-if capability. Adjusting a single slider (e.g., increasing stakeholder engagement) changed the prediction, and the explanation dynamically updated to reflect the reduced contribution of that feature. This interactivity supports risk mitigation planning.

## 6.5 Objective 4: Artefact Evaluation

The fourth objective was to evaluate the developed DSS artefact through structured system testing. The main results are drawn from the testing activities reported in Chapter Five.

**Functional correctness.** All unit and integration tests passed, and end-to-end scenarios completed without deviation from expected behaviour. The confusion matrices confirmed that 50.7% of test-set projects were correctly classified by Random Forest, with the majority of errors occurring between adjacent risk levels.

**Performance.** The API achieved an average response time of approximately 180 ms for `/predict` and 420 ms for `/explain`, both well within the 2-second target. 100% of requests met the benchmark. Batch prediction of 600 records completed in under 2 seconds using the optimised `predict_batch()` function.

**Explanation quality.** Explanations were assessed for fidelity, comprehensibility, and consistency. SHAP values were verified to handle edge cases gracefully. The natural-language renderings were reviewed by the project team and judged to be clear, relevant, and directly linked to actionable project parameters.

**Defect resolution.** Five defects were logged and resolved during the testing phase:
1. SHAP index out of bounds error
2. Slider type mismatch
3. Bar chart color length error
4. Categorical string to float error
5. Bold text asterisks

All defects were closed, and the system was verified to function correctly.

## 6.6 Comparative Evaluation Against Literature

A brief comparative analysis positions the developed DSS against systems and approaches identified in the literature review.

**Predictive performance.** The Random Forest model's accuracy (50.7%) and macro F1 (0.508) on a four-class ordinal problem are moderate, reflecting the complexity of the synthetic dataset. The quadratic-weighted Kappa of 0.666 (Logistic Regression) provides ordinal discrimination evidence. The within-one accuracy of 92.7% (Random Forest) indicates that the model rarely makes severe misclassifications, which is operationally important for risk assessment.

**Explainability integration.** Unlike systems that present SHAP outputs as static plots, the present DSS embeds explanations directly into the interactive dashboard, linking feature contributions to natural-language sentences and what-if controls. The implementation includes robust error handling for edge cases in SHAP computation.

**Architectural completeness.** The study delivers a working three-tier system with decoupled API and frontend, demonstrating feasibility beyond theoretical design. The use of Flask and Streamlit mirrors the rapid prototyping approach recommended in the literature, while maintaining the modularity stressed by software engineering best practices.

**Evaluation scope.** The present study's evaluation framework—spanning automated tests, performance timing, and qualitative explanation inspection—provides a comprehensive assessment of the integrated system. However, the absence of user-based usability testing remains a shared limitation with most existing studies.

## 6.7 Chapter Summary

This chapter presented results structured by the four project objectives. Data engineering produced a validated, clean dataset with 18 theoretically meaningful features. Model development delivered a Random Forest classifier with 50.7% accuracy and a quadratic weighted Kappa of 0.622, accompanied by SHAP explanations that reveal the key drivers of risk. The DSS artefact was realised as a fully functional three-tier web application with an interactive dashboard that supports prediction, explanation, and what-if analysis. A new Data Upload feature enables batch processing of large datasets. Structured evaluation confirmed functional correctness, swift response times, and high explanation quality. Five defects were identified and resolved during testing. These results provide the foundation for the discussion of implications and limitations in Chapter Seven.

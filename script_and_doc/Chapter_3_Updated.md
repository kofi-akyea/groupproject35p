# Chapter Three: Research Design

## 3.1 Introduction

This chapter presents the research design governing the development and evaluation of the web-based intelligent Decision Support System (DSS) for project risk prediction. It describes the research philosophy, Design Science Research (DSR) methodology, dataset and validation procedures, data preparation pipeline, machine learning and explainability implementation, system evaluation framework, and measures adopted to support validity, reliability, and ethical integrity. The research design is aligned with the artefact developed in this study, ensuring that the methodological decisions described in this chapter correspond directly to the implemented system and its subsequent evaluation.

## 3.2 Research Philosophy

The study adopts a pragmatic research philosophy, selecting methods according to their fitness for addressing the research problem and producing useful outcomes (Creswell, 2014). Pragmatism is appropriate because the study is primarily concerned with designing, implementing, and evaluating a working IT artefact rather than testing a purely theoretical proposition.

The research consequently combines quantitative and qualitative forms of assessment. Quantitative evidence is obtained through predictive-performance and system-performance measures, while qualitative assessment is applied to the interpretability and presentation of model explanations. This combination supports evaluation of both the computational behaviour of the artefact and the usefulness of its outputs within a decision-support context.

## 3.3 Design Science Research Methodology

The study employs Design Science Research (DSR), a methodology concerned with the construction and evaluation of artefacts designed to address identified problems (Hevner et al., 2004). DSR is appropriate because the principal outcome of this research is an operational web-based DSS rather than a theoretical model alone.

The DSR process was operationalised through three interconnected cycles:

- **Relevance Cycle:** The project-risk prediction problem and system requirements were established from the identified research problem and literature.
- **Design Cycle:** The DSS was developed through the integration of data preparation, machine learning, explainability, and web-based system components.
- **Rigour Cycle:** The resulting artefact was evaluated using predictive, functional, performance, and explanation-quality criteria.

The cycles were treated as interconnected rather than strictly linear. Development and evaluation activities informed refinement of the artefact, allowing implementation issues to be identified and addressed before final assessment. This provides methodological alignment between the research problem, artefact construction, and evaluation.

## 3.4 Dataset Description and Validation

The study uses the Project Management Risk Raw dataset obtained from Kaggle (ka66ledata, 2024). The dataset contains 4,000 project records and 51 features representing multiple dimensions of project risk.

Exploratory analysis identified a four-level ordinal target variable, Risk_Level, consisting of:

| Risk Level | Proportion |
|-----------|------------|
| Low | 20.15% |
| Medium | 34.90% |
| High | 25.90% |
| Critical | 19.05% |

The four-level taxonomy was retained rather than collapsing Critical into High because the ordering represents different levels of risk severity and the Critical category provides a distinct operational classification for the decision-support system. Maintaining the four categories also ensures that the implemented artefact reflects the structure present in the source dataset rather than simplifying the prediction task without justification.

Data validation identified missing values in Risk_Management_Maturity, Change_Control_Maturity, and Tech_Environment_Stability. The validation process also identified out-of-range values in Communication_Frequency (51 values above the maximum of 10.0). These findings informed the subsequent preprocessing and data-quality procedures rather than being treated as grounds for indiscriminate record removal.

A chi-square test of independence was also conducted to examine associations between the categorical predictor information and the Risk_Level outcome. The resulting significance level of p < 0.001 provided statistical evidence of an association between the examined variables and risk classification. This result was treated as supporting evidence for proceeding with predictive modelling; it was not interpreted as evidence that the machine learning models would necessarily achieve high predictive performance.

## 3.5 Data Preparation Pipeline

A reproducible data-preparation pipeline was implemented to transform the raw dataset into the feature representation required by the machine learning models. The preprocessing configuration was centralised in config.py, providing a consistent definition of the features used by the application and reducing the possibility of discrepancies between model development and system deployment.

### 3.5.1 Missing-Value Treatment

Missing values were handled according to variable type. Missing continuous values were imputed using the median, while missing categorical values were imputed using the mode. This approach allowed incomplete observations to be retained while using robust summary statistics appropriate to the respective variable types.

The identified data-quality issues were therefore incorporated into a defined preprocessing process rather than handled manually on an individual-record basis.

### 3.5.2 Stakeholder Engagement Transformation

A specific transformation was applied to Stakeholder_Engagement_Level because the variable represents an ordered progression of stakeholder engagement. The implementation converted the original categorical levels into a numerical representation using the following mapping:

| Stakeholder Engagement | Numerical Value |
|----------------------|-----------------|
| Poor | 0.00 |
| Low | 0.25 |
| Medium | 0.50 |
| High | 0.75 |
| Excellent | 1.00 |

This transformation preserves the relative ordering of the categories while providing a numerical representation suitable for model processing. The mapping was explicitly defined rather than relying on arbitrary category encoding, thereby maintaining the intended semantic ordering of stakeholder engagement.

### 3.5.3 Feature Configuration and Final Feature Set

Feature definitions were maintained centrally through config.py. The final feature set comprising 18 predictors was selected based on domain knowledge and project management literature, representing the most relevant dimensions of project risk across six categories: Project characteristics, Team factors, Governance, and Schedule/Budget/External factors. The complete predictor list is presented in Table 6.1 in the evaluation/results chapters.

The 18 features were defined as follows:

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

Centralising the feature configuration provides traceability between the trained models and the deployed DSS. It also reduces the risk that the application accepts or processes a feature representation that differs from the representation used during model development.

### 3.5.4 Dataset Partitioning

The 4,000 observations were partitioned using stratified sampling into three subsets:

| Dataset | Records | Proportion |
|---------|---------|------------|
| Training | 2,800 | 70% |
| Validation | 600 | 15% |
| Testing | 600 | 15% |
| Total | 4,000 | 100% |

The training set was used for model development, the validation set supported model assessment during development, and the test set was reserved for final evaluation. Stratification was used to maintain representation of the four Risk_Level categories across the partitions.

## 3.6 Machine Learning and Explainability

### 3.6.1 Machine Learning Models

Two complementary supervised learning approaches were implemented: Multinomial Logistic Regression and Random Forest.

Multinomial Logistic Regression provides a comparatively transparent statistical baseline for multiclass prediction, while Random Forest provides an ensemble-based approach capable of modelling nonlinear relationships and interactions among predictors. Using both models enables comparison between a relatively interpretable baseline and a more flexible ensemble approach.

The models were trained using the implemented 18-predictor feature set. Their performance was subsequently assessed using both conventional classification measures and metrics reflecting the ordered nature of the risk categories.

Importantly, although Risk_Level is ordinal, the implemented models perform multiclass classification rather than specialised ordinal regression. The ordinal structure is therefore reflected explicitly in the evaluation strategy and interpretation of classification errors rather than being represented as a dedicated ordinal-learning objective within the two selected algorithms.

### 3.6.2 Ordinal-Aware Evaluation

Because the target contains the ordered categories Low, Medium, High, and Critical, model assessment was extended beyond conventional accuracy.

The evaluation included quadratic-weighted Cohen's Kappa, which gives greater consideration to the distance between observed and predicted categories. This is relevant to the project because an error between adjacent categories, such as Medium and High, is qualitatively different from an error between Low and Critical.

The evaluation also included within-one accuracy. This measure identifies predictions that fall within one risk category of the correct classification. For example, predicting High when the actual category is Medium is counted as within one category, whereas predicting Critical for a Low-risk project is not. This provides an additional view of model usefulness where near-miss classifications may be operationally more acceptable than substantially distant errors.

The combination of conventional classification measures with ordinal-aware measures therefore provides a more appropriate assessment of the four-level risk classification task than accuracy alone.

### 3.6.3 Explainable AI Implementation

Explainable Artificial Intelligence was incorporated using SHAP (SHapley Additive exPlanations). For the Random Forest model, explanations were generated using TreeExplainer, allowing feature contributions to be calculated for individual predictions and used within the DSS.

The explainability implementation supports two principal levels of interpretation:

- **Global interpretation**, identifying features that have greater influence across model predictions.
- **Local interpretation**, identifying the feature contributions associated with an individual risk prediction.

The explanation component was integrated into the DSS rather than being treated as a separate offline analysis. This allows the prediction and its associated explanatory information to be presented as part of the same decision-support workflow.

The implementation also incorporated robust handling of empty SHAP arrays. This prevents explanation-generation failures in cases where the returned explanation structure does not contain the expected values and improves the stability of the explanation component during system operation, particularly during batch predictions.

## 3.7 System Evaluation Framework

The evaluation framework was designed to assess the implemented DSS at multiple levels rather than evaluating predictive accuracy in isolation. Four principal dimensions were considered.

### 3.7.1 Predictive Performance

The machine learning component was evaluated using classification performance measures together with ordinal-aware measures. The latter are particularly important because the four Risk_Level categories have an inherent order. The evaluation therefore considers not only whether a prediction is correct, but also how far an incorrect prediction is from the actual category.

### 3.7.2 Functional Testing

Functional testing was used to determine whether the implemented DSS performed its intended functions correctly. Testing covered the principal workflow from user input through prediction and explanation generation to presentation of the resulting output.

The objective was to verify that the implemented components operated according to their specified requirements and that the integration between the machine learning and web application components functioned correctly.

### 3.7.3 System Performance

System performance was assessed to determine whether the implemented DSS could generate predictions and associated outputs within an acceptable response period. Performance assessment considered the behaviour of the integrated system rather than the machine learning model in isolation.

### 3.7.4 Explanation Quality

Explanation quality was evaluated as a distinct component because predictive performance alone does not establish that a model is understandable. The assessment considered whether the SHAP-based outputs corresponded to the model prediction and whether the resulting information could be presented in a meaningful form to DSS users.

The explanation component was therefore evaluated both technically, through the generated SHAP information, and functionally, through its successful integration into the DSS interface.

## 3.8 Validity, Reliability, and Reproducibility

Several measures were incorporated to strengthen the methodological quality of the study.

Validity was supported through the use of a defined four-level target, explicit data-validation procedures, stratified dataset partitioning, and evaluation using measures appropriate to both multiclass and ordinal classification.

Reliability was supported through the reproducible preprocessing pipeline and centralised feature configuration. The use of config.py provides a consistent feature definition across the machine learning and application components.

Reproducibility was further supported by documenting the implemented preprocessing transformations, feature representation, dataset partition sizes, machine learning approaches, and explainability mechanism. These details provide a clear basis for reproducing the implemented workflow using the same dataset and configuration.

The study nevertheless recognises that the use of a single synthetic dataset limits the extent to which the findings can be generalised to real-world project environments. Consequently, the evaluation primarily establishes the technical feasibility and behaviour of the developed artefact rather than claiming validated real-world predictive effectiveness.

## 3.9 Ethical Considerations

The study uses the Project Management Risk Raw dataset as its data source and does not involve the collection of personal information from human participants. The DSS is designed as a decision-support tool rather than an autonomous decision-maker. Its predictions and explanations are intended to support human judgement rather than replace professional project-risk assessment.

The inclusion of explainability further supports responsible use by providing information concerning the factors contributing to model predictions. This enables users to scrutinise outputs rather than receiving risk classifications without supporting information.

## 3.10 Summary

This chapter established the research design underpinning the development and evaluation of the web-based intelligent DSS. A pragmatic philosophy and Design Science Research methodology were adopted to support the construction and evaluation of the artefact. The study used a 4,000-record dataset containing 51 features and retained its four-level Risk_Level taxonomy of Low, Medium, High, and Critical.

The implemented preprocessing pipeline addressed identified data-quality issues, applied defined missing-value treatments, transformed Stakeholder_Engagement_Level using an explicit ordered mapping, centralised feature definitions through config.py, and produced a final set of 18 predictors selected based on domain knowledge and project management literature. The dataset was partitioned into 2,800 training, 600 validation, and 600 testing records using stratified sampling.

Multinomial Logistic Regression and Random Forest were implemented as complementary multiclass approaches. Evaluation incorporated ordinal-aware measures, including quadratic-weighted Cohen's Kappa and within-one accuracy, while SHAP with TreeExplainer provided model explanations for the Random Forest component. The overall evaluation framework covered predictive performance, functional correctness, system performance, and explanation quality. These methodological decisions establish a direct connection between the implemented artefact and the evaluation presented in the subsequent chapters.

## References

- Creswell, J.W. (2014) *Research Design: Qualitative, Quantitative, and Mixed Methods Approaches*. 4th edn. Thousand Oaks, CA: Sage.
- Gregor, S. and Hevner, A.R. (2013) 'Positioning and presenting design science research for maximum impact', *MIS Quarterly*, 37(2), pp. 337–355.
- Haixiang, G. et al. (2021) 'Impact of dataset size on classification performance', *Applied Sciences*, 11(2), p. 796.
- Hevner, A.R., March, S.T., Park, J. and Ram, S. (2004) 'Design science in information systems research', *MIS Quarterly*, 28(1), pp. 75–105.
- Hillson, D. (2009) *Managing Risk in Projects*. Farnham: Gower Publishing.
- ka66ledata (2024) *Project management Risk Raw Dataset*. Kaggle. Available at: https://www.kaggle.com/datasets/ka66ledata/project-management-risk-raw (Accessed: 18 May 2026).
- Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems 30*, pp. 4765–4774.
- Mohseni, S., Zarei, N. and Ragan, E.D. (2021) 'A multidisciplinary survey and framework for design and evaluation of explainable AI systems', *ACM Transactions on Interactive Intelligent Systems*, 11(3–4), pp. 1–45.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.
- Project Management Institute (2021) *A Guide to the Project Management Body of Knowledge (PMBOK® Guide)*. 7th edn. Newtown Square, PA: PMI.

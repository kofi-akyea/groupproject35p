# Defense Guide: Samuel Kwesi Bonku - Data Engineering

## Role Overview

**Primary Responsibility:** Data Engineering and Preprocessing
**Chapter Reference:** Chapter Four, Section 4.4 - Data Preprocessing Implementation
**Key Deliverables:** Clean, validated, and preprocessed dataset ready for machine learning

---

## 1. My Contribution Summary

### Core Responsibilities
- **Data Loading and Validation:** Implemented robust data loading with comprehensive validation checks
- **Data Cleaning:** Developed cleaning pipeline handling missing values, type conversions, and range validation
- **Feature Engineering:** Transformed Stakeholder_Engagement_Level from categorical to numeric with semantic preservation
- **Dataset Partitioning:** Implemented stratified sampling to maintain class distribution across train/val/test splits
- **Configuration Management:** Centralised all feature specifications in `config.py` as single source of truth

### Key Files Implemented
- `code/data_prep/load_data.py` - Data loading from CSV and parquet formats
- `code/data_prep/validate_data.py` - Data quality validation and range checking
- `code/data_prep/clean_data.py` - Missing value imputation, type conversion, and clipping
- `code/data_prep/encode_features.py` - Ordinal and one-hot encoding
- `code/data_prep/split_data.py` - Stratified train/val/test split
- `code/utils/config.py` - Centralised feature configuration (18 features)

---

## 2. Technical Decisions and Rationale

### Decision 1: Median Imputation for Continuous Variables
**What I did:** Used median instead of mean for imputing missing continuous values
**Why:** Median is more robust to outliers and extreme values, which are common in project risk data
**Evidence:** Implemented in `clean_data.py` line 45-50

### Decision 2: Mode Imputation for Categorical Variables
**What I did:** Used mode (most frequent value) for missing categorical data
**Why:** Preserves the most common category and avoids introducing artificial categories
**Evidence:** Implemented in `clean_data.py` line 52-57

### Decision 3: Stakeholder_Engagement_Level Ordinal Transformation
**What I did:** Converted categorical levels to numeric values (Poor=0.0, Low=0.25, Medium=0.5, High=0.75, Excellent=1.0)
**Why:** Preserves the semantic ordering while providing numerical representation for ML algorithms
**Evidence:** Implemented in `clean_data.py` line 30-38 with explicit mapping

### Decision 4: Centralised Configuration
**What I did:** Created `config.py` as single source of truth for all feature specifications
**Why:** Ensures consistency between training and inference, prevents feature mismatches
**Evidence:** `code/utils/config.py` defines all 18 features with types, ranges, and defaults

### Decision 5: Stratified Sampling
**What I did:** Used stratified sampling for train/val/test split (70%/15%/15%)
**Why:** Maintains the four-class distribution across all splits, preventing bias
**Evidence:** Implemented in `split_data.py` line 15-25

---

## 3. Challenges Faced and Solutions

### Challenge 1: Stakeholder_Engagement_Level Type Conversion
**Problem:** Original categorical values caused type conversion errors during model inference
**Solution:** Implemented explicit ordinal mapping with semantic preservation
**Learning:** Data type consistency is critical across the entire pipeline

### Challenge 2: Out-of-Range Communication_Frequency Values
**Problem:** 51 values exceeded the maximum of 10.0
**Solution:** Implemented clipping to valid range with validation warning
**Learning:** Data validation must catch edge cases before they affect models

### Challenge 3: Feature Name Consistency
**Problem:** Different naming conventions between preprocessing and inference
**Solution:** Centralised all feature names in `config.py` as single source of truth
**Learning:** Configuration management prevents integration bugs

---

## 4. Code Evidence

### Data Loading Implementation
```python
# code/data_prep/load_data.py
def load_raw(path: Path = None) -> pd.DataFrame:
    """Load raw CSV dataset."""
    if path is None:
        path = RAW_DIR / "project_risk_raw_dataset.csv"
    return pd.read_csv(path)
```

### Stakeholder_Engagement_Level Transformation
```python
# code/data_prep/clean_data.py
SE_LEVEL_MAP = {"Low": 0.25, "Medium": 0.50, "High": 0.75, "Excellent": 1.0, "Poor": 0.0}

if "Stakeholder_Engagement_Level" in df.columns:
    df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].astype(str)
    df["Stakeholder_Engagement_Level"] = df["Stakeholder_Engagement_Level"].map(SE_LEVEL_MAP).fillna(0.5)
    df["Stakeholder_Engagement_Level"] = pd.to_numeric(df["Stakeholder_Engagement_Level"], errors="coerce")
```

### Centralised Feature Configuration
```python
# code/utils/config.py
FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("Project_Type", "Project Type", "nominal", ...),
    FeatureSpec("Complexity_Score", "Complexity Score", "numeric", ...),
    # ... 16 more features
]
FEATURE_NAMES = [f.name for f in FEATURE_SPECS]
```

---

## 5. Defense Strategy

### Key Points to Emphasize
1. **Reproducibility:** My pipeline is fully reproducible with clear documentation
2. **Robustness:** Handles missing values, outliers, and type mismatches gracefully
3. **Consistency:** Centralised configuration ensures no feature mismatches
4. **Domain Knowledge:** Feature selection based on project management literature
5. **Data Quality:** Comprehensive validation catches issues before model training

### Potential Questions and Answers

**Q: Why did you choose median over mean for imputation?**
A: Median is more robust to outliers. Project risk data often has extreme values (e.g., very high budget overruns), and mean would be skewed by these outliers, leading to biased imputations.

**Q: How did you ensure no data leakage between train and test sets?**
A: I implemented stratified sampling in the split function, which maintains the class distribution while keeping the sets completely separate. All preprocessing transformations are learned from the training set only.

**Q: Why transform Stakeholder_Engagement_Level to numeric?**
A: The original categorical values had a clear ordinal progression. Converting to numeric preserves this ordering while making the feature compatible with ML algorithms that require numeric inputs.

**Q: What would you do differently if you had more time?**
A: I would implement more sophisticated imputation strategies (e.g., KNN imputation) and conduct more extensive exploratory data analysis to identify additional feature engineering opportunities.

---

## 6. Test Evidence

### Unit Tests Passed
- `test_load_raw_ok` - Data loading functionality
- `test_validate_catches_missing_required_field` - Validation error detection
- `test_clean_returns_four_levels` - Cleaning preserves risk levels
- `test_clean_critical_count_at_least_one` - Critical class preserved
- `test_encode_target_ordinal` - Ordinal encoding correctness
- `test_split_preserves_class_proportions` - Stratified split maintains distribution
- `test_build_preprocessor_fits` - Preprocessor fitting works correctly

All 7 data preparation tests passed successfully.

---

## 7. Integration with Other Components

### How My Work Connects to Sylvester (Model Development)
- Provides clean, validated dataset for model training
- Ensures feature types match model requirements
- Centralised configuration prevents feature mismatches

### How My Work Connects to Kofi (Web Application)
- Feature specifications used in dashboard input forms
- Data cleaning logic applied to user-uploaded data
- Configuration ensures UI labels match feature names

### How My Work Connects to Festus (Testing)
- Test data split used for model evaluation
- Validation checks tested for error handling
- Reproducible pipeline enables systematic testing

---

## 8. Key Metrics and Results

### Dataset Statistics
- **Total Records:** 4,000
- **Features:** 51 original → 18 selected
- **Target Classes:** 4 (Low, Medium, High, Critical)
- **Missing Values Handled:** NaN in 3 governance columns
- **Out-of-Range Values:** 51 Communication_Frequency values clipped

### Data Split Distribution
- **Training:** 2,800 records (70%)
- **Validation:** 600 records (15%)
- **Testing:** 600 records (15%)
- **Class Distribution Preserved:** Within ±0.3% across all splits

---

## 9. References to Literature

My approach aligns with:
- **Hillson (2009):** Project risk management best practices
- **PMI (2021):** PMBOK Guide data quality standards
- **Pedregosa et al. (2011):** scikit-learn preprocessing best practices

---

## 10. Closing Statement

My data engineering work established a solid foundation for the entire DSS. By implementing a robust, reproducible preprocessing pipeline with centralised configuration, I ensured that the models, web application, and testing framework all operate on consistent, high-quality data. The challenges I faced taught me the importance of data type consistency, configuration management, and comprehensive validation in machine learning pipelines.

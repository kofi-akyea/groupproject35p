# Project Risk Decision Support System (DSS)

A comprehensive machine learning system for predicting project risk levels with explainable AI. The system uses a four-level ordinal classification (Low, Medium, High, Critical) and provides SHAP-based explanations for predictions.

## 🎯 Project Overview

This DSS helps project managers and stakeholders assess project risk by analyzing 18 key features across project characteristics, team composition, and environmental factors. The system includes:

- **Data Pipeline**: Automated data loading, cleaning, validation, and preprocessing
- **Model Training**: 6 models - Logistic Regression, Random Forest, Ordinal Logistic Regression, XGBoost, SVM (RBF kernel), and K-Nearest Neighbours (KNN) with ordinal evaluation
- **Explainability**: SHAP-based local and global explanations with natural language narratives
- **REST API**: Flask API for programmatic access to predictions
- **Interactive Dashboard**: Streamlit web interface for risk prediction and analytics

## 📊 Dataset

- **Source**: Project Management Risk Raw Dataset (synthetic, 4000 records)
- **Target**: Risk_Level (4 ordinal levels: Low, Medium, High, Critical)
- **Features**: 18 features including project type, complexity score, methodology, team experience, resource availability, etc.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the System

1. **Train Models** (run first):
```bash
python pipeline.py
```
This will:
- Load and clean the raw dataset
- Split into train/val/test sets
- Train 6 models: Logistic Regression, Random Forest, Ordinal Logistic Regression, XGBoost, SVM (RBF kernel), and K-Nearest Neighbours (KNN)
- Evaluate all models and save reports

2. **Launch Dashboard**:
```bash
streamlit run dashboard/app.py
```
Open your browser to `http://localhost:8501`

3. **Start API Server**:
```bash
python -m code.api.app
```
API will be available at `http://localhost:5000`

## 📁 Project Structure

```
DSS SYSTEM GROUP/
├── 📄 pipeline.py                    # Main training pipeline
├── 📄 requirements.txt               # Python dependencies
├── 📄 project_risk_raw_dataset.csv   # Raw dataset
├── 📄 PROJECT_STRUCTURE.md           # Detailed folder structure guide
│
├── 📁 data/                          # Data storage
│   ├── raw/                          # Original unprocessed data
│   └── processed/                    # Cleaned and split datasets
│
├── 📁 models/                        # Trained model files
│   ├── logistic_regression.joblib   # Logistic Regression model
│   ├── random_forest.joblib          # Random Forest model
│   ├── ordinal_logistic_regression.joblib  # Ordinal Logistic Regression model
│   ├── xgboost.joblib                # XGBoost model
│   ├── svm_rbf.joblib                # SVM (RBF kernel) model
│   ├── knn.joblib                    # K-Nearest Neighbours model
│   └── model_card.json               # Model metadata
│
├── 📁 reports/                       # Evaluation reports
│
├── 📁 dashboard/                     # Streamlit web interface
│   ├── app.py                        # Main dashboard with top navigation
│   ├── theme.py                      # Theme configuration and styling
│   └── components/                   # Reusable UI components
│
├── 📁 code/                          # Source code modules
│   ├── data_prep/                    # Data preprocessing
│   ├── models/                       # Model training
│   ├── explainability/               # Model explanations
│   ├── api/                          # REST API
│   └── utils/                        # Utilities & config
│
├── 📁 tests/                         # Test modules

```

## 🔑 Key Features

### 1. Four-Level Ordinal Risk Classification
- **Low** (Green): Minimal risk, standard monitoring
- **Medium** (Yellow): Moderate risk, increased attention
- **High** (Orange): Significant risk, mitigation required
- **Critical** (Red): Severe risk, immediate action needed

### 2. Ordinal-Aware Evaluation
- **Quadratic Weighted Kappa (QWK)**: Measures agreement accounting for ordinal nature
- **Within-One Accuracy**: Allows predictions within one ordinal level
- **Macro F1**: Balanced performance across classes

### 3. Explainable AI
- **SHAP Values**: Feature importance for individual predictions (available for Random Forest and XGBoost models only)
- **Natural Language Narratives**: Plain-English explanations for tree-based model predictions (Random Forest and XGBoost)
- **Waterfall Plots**: Visual explanation of feature contributions (Random Forest and XGBoost)
- **Note**: For Multinomial Logistic Regression, Ordinal Logistic Regression, SVM, and KNN models, the system provides predictions and probabilities without feature attribution explanations

## 📈 Model Performance

### How to Determine the Best Model

For ordinal classification problems (Low, Medium, High, Critical), **no single metric tells the whole story**. Use this decision framework:

**Primary Metric: QWK (Quadratic Weighted Kappa)**
- **Why**: The gold standard for ordinal problems - accounts for the ordered nature of classes
- **Interpretation**: Measures agreement between predictions and actual values, weighted by ordinal distance
- **Higher is better**: 0.68+ is good, 0.70+ is excellent

**Secondary Metric: Within-One Accuracy**
- **Why**: More lenient for ordinal problems - counts predictions within ±1 level as correct
- **Interpretation**: Shows how often the model is "close enough" for practical decision-making
- **Higher is better**: 0.90+ indicates the model rarely makes severe errors

**Tertiary Metric: Exact-Match Accuracy**
- **Why**: Traditional accuracy - only counts exact matches
- **Interpretation**: Strict measure of perfect predictions
- **Higher is better**: But less informative for ordinal problems

**Support Metric: Macro F2**
- **Why**: Prioritizes recall over precision - critical for risk detection where missing high-risk cases is costly
- **Interpretation**: F-beta score with β=2, weighting recall twice as heavily as precision
- **Formula**: F₂ = (1 + 2²) × (precision × recall) / (2² × precision + recall) = 5 × (precision × recall) / (4 × precision + recall)
- **Higher is better**: 0.50+ indicates reasonable balance with emphasis on catching high-risk cases

### Model Selection Recommendation

**Best Overall Model: Ordinal Logistic Regression**
- Highest QWK (0.6829 → 68.29%) - best ordinal agreement
- Highest Within-One Accuracy (0.9617 → 96.17%) - rarely makes severe errors
- Strong Macro F2 (0.5082 → 50.82%) - prioritizes capturing high-risk cases
- **Use for**: Production deployment when ordinal accuracy and risk detection matter most

**Alternative: Random Forest**
- Highest Exact-Match Accuracy (0.5217 → 52.17%) - best strict accuracy
- Strong Within-One Accuracy (0.9483 → 94.83%) - good ordinal performance
- Good Macro F2 (0.5081 → 50.81%) - prioritizes capturing high-risk cases
- **Use for**: When exact matches are critical and robust ensemble performance is desired

**Highest Macro F2: Logistic Regression**
- Highest Macro F2 (0.5241 → 52.41%) - best at prioritizing recall for risk detection
- Strong QWK (0.6604 → 66.04%) - good ordinal agreement
- **Use for**: When capturing high-risk cases is the top priority

**Percentage Conversion Formula**
```
Percentage = Decimal Value × 100
Example: 0.6829 × 100 = 68.29%
```

### Current Model Performance

After running the pipeline, you'll see evaluation metrics for all 6 models:

```
Logistic Regression:
  Exact-Match Accuracy: 0.4950, QWK: 0.6604, Within-One: 0.9067, Macro F2: 0.5241

Random Forest:
  Exact-Match Accuracy: 0.5217, QWK: 0.6481, Within-One: 0.9483, Macro F2: 0.5081

Ordinal Logistic Regression:
  Exact-Match Accuracy: 0.5067, QWK: 0.6829, Within-One: 0.9617, Macro F2: 0.5082

XGBoost:
  Exact-Match Accuracy: 0.4900, QWK: 0.6239, Within-One: 0.9283, Macro F2: 0.4904

SVM (RBF Kernel):
  Exact-Match Accuracy: 0.4933, QWK: 0.6317, Within-One: 0.9483, Macro F2: 0.4779

K-Nearest Neighbors (KNN):
  Exact-Match Accuracy: 0.4117, QWK: 0.4948, Within-One: 0.8833, Macro F2: 0.4071
```

### Model Prediction Formulas

Each model uses a different mathematical approach to generate predictions:

**Logistic Regression:**
```
P(Y=k|X) = softmax(β_k · X + b_k)
Prediction: argmax_k P(Y=k|X)
```

**Random Forest:**
```
Prediction: majority_vote{f_t(X) for t in 1..500}
Probability: (votes_for_class_k) / 500
```

**Ordinal Logistic Regression:**
```
P(Y ≤ k|X) = σ(θ_k - β·X) where σ(z) = 1/(1+e^(-z))
P(Y=k|X) = P(Y ≤ k|X) - P(Y ≤ k-1|X)
```

**XGBoost:**
```
ŷ = Σ f_t(X) where f_t are gradient-boosted trees
P(Y=k|X) = softmax(ŷ_k) = exp(ŷ_k) / Σ_j exp(ŷ_j)
```

**SVM (RBF Kernel):**
```
f(x) = Σ α_i y_i K(x_i, x) + b
K(x_i, x) = exp(-γ ||x_i - x||^2)
Probability: Platt scaling applied to decision function
```

### Probability to Percentage Conversion

The API converts model probabilities from decimal (0.0-1.0) to percentage (0.0-100.0%) for user-friendly display:

```
percentage = round(probability × 100, 2)
```

Example:
- Raw probability: 0.452
- Displayed percentage: 45.2%

**Important Distinction: Overall Metrics vs. Single Prediction Probabilities**

There are two different percentage concepts in model evaluation:

**1. Overall Model Performance Metrics (Aggregate across test set):**
These metrics represent the model's overall accuracy across all 600 test samples. They are NOT the probability of a single prediction.

- **Accuracy (e.g., 52.17%):** Percentage of correct predictions out of 600 total predictions
  - Formula: `Accuracy = (Correct Predictions / Total Predictions) × 100`
  - Example: 313 correct out of 600 = 52.17%

- **Macro F1 (e.g., 50.8%):** Average of F1 scores across all 4 classes
  - Formula: `Macro F1 = (F1_Low + F1_Medium + F1_High + F1_Critical) / 4 × 100`

- **QWK (e.g., 68.3%):** Quadratic Weighted Kappa - ordinal agreement metric
  - Measures agreement between predicted and actual classes, weighted by ordinal distance

- **Within-one (e.g., 96.2%):** Percentage of predictions within one class of actual
  - Formula: `Within-one = (Predictions within ±1 class / Total Predictions) × 100`

**2. Single Prediction Probabilities (Per-sample class probabilities):**
For each individual project, the model outputs a probability distribution across the 4 risk levels. These are converted to percentages for user-friendly display.

- Formula: `Class_Percentage = round(Class_Probability × 100, 2)`
- Example for a single project:
  - Low: 0.15 → 15.00%
  - Medium: 0.45 → 45.00%
  - High: 0.30 → 30.00%
  - Critical: 0.10 → 10.00%
  - Sum: 100.00%

**Key Difference:**
- Overall metrics (50.7% accuracy) describe the model's performance across ALL test samples
- Single prediction percentages (15%, 45%, 30%, 10%) describe the confidence distribution for ONE specific project

## 🔧 Configuration

All feature specifications and paths are configured in:
- `code/utils/config.py`: Feature definitions, data paths, model paths
- `code/utils/risk_levels.py`: Risk level colors, actions, and mappings

## 📡 API Usage

### Predict Endpoint

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Project_Type": "IT",
    "Complexity_Score": 7.5,
    "Methodology_Used": "Agile",
    "Project_Phase": "Execution",
    "Team_Experience_Level": "Mixed",
    "Project_Manager_Experience": "Mid-level PM",
    "Resource_Availability": 0.6,
    "Team_Turnover_Rate": 0.2,
    "Requirement_Stability": "Moderate",
    "Risk_Management_Maturity": "Formal",
    "Change_Control_Maturity": "Formal",
    "Communication_Frequency": 4.0,
    "Stakeholder_Engagement_Level": 0.7,
    "Schedule_Pressure": 0.5,
    "Budget_Utilization_Rate": 0.9,
    "Historical_Risk_Incidents": 3,
    "Vendor_Reliability_Score": 0.7,
    "Tech_Environment_Stability": "Mixed"
  }'
```

### Response

```json
{
  "request_id": "uuid",
  "prediction": "High",
  "probabilities": {
    "Low": 10.0,
    "Medium": 30.0,
    "High": 40.0,
    "Critical": 20.0
  },
  "shap": {
    "Complexity_Score": 0.25,
    "Team_Turnover_Rate": 0.15,
    ...
  },
  "top_features": [
    {"feature": "Complexity_Score", "shap": 0.25},
    ...
  ],
  "narrative": "**Predicted Risk Level**: High\n\n...",
  "model_version": "1.0.0"
}
```

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests (with debugging plugin disabled to avoid stdlib 'code' module conflict)
python -m pytest tests/ -v -p no:debugging

# Run specific test file
python -m pytest tests/test_data_prep.py -v -p no:debugging
python -m pytest tests/test_models.py -v -p no:debugging

# Run with coverage
python -m pytest tests/ -v -p no:debugging --cov=code --cov-report=html
```

### Test Coverage

The test suite includes:

**Data Preparation Tests (7 tests):**
- Data loading and validation
- Data cleaning and type conversion
- Ordinal target encoding
- Stratified dataset splitting
- Preprocessor fitting and transformation

**Model Tests (6 tests):**
- Logistic Regression performance validation
- Random Forest performance validation
- Quadratic Weighted Kappa metric validation
- Within-one accuracy metric validation
- Ordinal metric edge cases

### Expected Output

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.0.2
collected 13 items

tests/test_data_prep.py::test_load_raw_ok PASSED
tests/test_data_prep.py::test_validate_catches_missing_required_field PASSED
tests/test_data_prep.py::test_clean_returns_four_levels PASSED
tests/test_data_prep.py::test_clean_critical_count_at_least_one PASSED
tests/test_data_prep.py::test_encode_target_ordinal PASSED
tests/test_data_prep.py::test_split_preserves_class_proportions PASSED
tests/test_data_prep.py::test_build_preprocessor_fits PASSED
tests/test_models.py::test_logreg_above_chance PASSED
tests/test_models.py::test_rf_above_chance PASSED
tests/test_models.py::test_qwk_perfect_agreement PASSED
tests/test_models.py::test_within_one_perfect PASSED
tests/test_models.py::test_within_one_off_by_one PASSED
tests/test_models.py::test_within_one_off_by_two PASSED

======================== 13 passed, 1 warning in 4.60s ========================
```


## � IT/DevOps Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Model Management
```bash
# Train models from scratch
python pipeline.py

# Check model metadata
python -c "import json; print(json.dumps(json.load(open('models/model_card.json')), indent=2))"

# Verify model files exist
python -c "import os; print('Models exist:', all(os.path.exists(f) for f in ['models/random_forest.joblib', 'models/logistic_regression.joblib']))"
```

### API Operations
```bash
# Start Flask API server
python -m code.api.app

# Check API health
curl http://localhost:5000/health

# Test prediction endpoint
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @test_payload.json
```

### Dashboard Operations
```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py

# Start dashboard on custom port
streamlit run dashboard/app.py --server.port 8503

# Clear Streamlit cache
streamlit cache clear
```

### Data Operations
```bash
# Verify dataset exists
python -c "import pandas as pd; print('Dataset shape:', pd.read_csv('project_risk_raw_dataset.csv').shape)"

# Check processed data
python -c "import pandas as pd; print('X_train shape:', pd.read_parquet('data/processed/X_train.parquet').shape)"
```

### Monitoring
```bash
# Check system resources (Windows)
wmic cpu get loadpercentage /value

# Check disk space
wmic logicaldisk get size,freespace,caption

# Check Python version
python --version

# Check installed packages
pip list
```

## �📚 Module Documentation

### Data Preprocessing (`code/data_prep/`)
- `load_data.py`: Load raw CSV and processed parquet files
- `validate_data.py`: Validate data quality and ranges
- `clean_data.py`: Handle missing values, convert types, clip ranges
- `encode_features.py`: One-hot and ordinal encoding
- `split_data.py`: Stratified train/val/test split
- `select_features.py`: L1 regularisation for feature selection diagnostics

### Model Training (`code/models/`)
- `train_logreg.py`: Logistic Regression training
- `train_rf.py`: Random Forest training
- `train_ordinal_logreg.py`: Ordinal Logistic Regression training
- `train_xgboost.py`: XGBoost training
- `train_svm.py`: SVM (RBF kernel) training
- `train_knn.py`: K-Nearest Neighbours training
- `evaluate.py`: Model evaluation with ordinal metrics
- `ordinal_metrics.py`: QWK, within-one accuracy, macro F2

### Explainability (`code/explainability/`)
- `shap_engine.py`: SHAP value computation for tree-based models (Random Forest and XGBoost)
- `narrative_builder.py`: Natural language explanation generation

### API (`code/api/`)
- `app.py`: Flask application factory
- `routes.py`: API endpoints (/predict, /explain, /health)
- `schemas.py`: Pydantic request/response schemas
- `inference.py`: Prediction inference logic
- `errors.py`: Error handling and validation

### Dashboard (`dashboard/`)
- `app.py`: Main Streamlit application with top navigation tabs (Home, Predictor, Analytics, Model Performance, Data Upload)
- `theme.py`: Theme configuration with Blue iris color palette and Apple-inspired design
- `components/`: Input form, risk gauge, probability bars, SHAP waterfall, NLG panel

## 🤝 Contributing

This is a research-aligned project. When making changes:
1. Ensure alignment with the four-level ordinal classification
2. Maintain the 18-feature specification in `code/utils/config.py`
3. Run the pipeline after changes to verify model performance
4. Update tests for new functionality

## 📄 License

This project is for educational and research purposes.

## 🆘 Troubleshooting

### Pipeline fails with "No module named 'src'"
- The folder was renamed from `src` to `code`. Ensure all imports use `code` instead of `src`.

### Stakeholder_Engagement_Level shows NaN values
- This column is converted from categorical to numeric during cleaning. Ensure the raw dataset has valid values.

### Dashboard shows "No trained models found"
- Run `python pipeline.py` first to train and save models.

### API returns 500 error
- Check that models exist in the `models/` directory
- Verify the API is running on the correct port (default 5000)

## 📞 Support

For detailed folder structure, see `PROJECT_STRUCTURE.md`.
# groupproject35p

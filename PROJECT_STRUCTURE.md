# Project Risk Decision Support System - Folder Structure

This document explains the organization of the project folders for easy navigation.

## 📁 Root Directory

```
DSS SYSTEM GROUP/
├── 📄 pipeline.py                    # Main training pipeline (run this first)
├── 📄 requirements.txt               # Python dependencies
├── 📄 project_risk_raw_dataset.csv   # Raw dataset (4000 records)
│
├── 📁 data/                          # Data storage
│   ├── raw/                          # Original unprocessed data
│   └── processed/                    # Cleaned and split datasets
│
├── 📁 models/                        # Trained model files
│   ├── logistic_regression.joblib   # Logistic Regression model
│   ├── random_forest.joblib          # Random Forest model
│   └── model_card.json               # Model metadata
│
├── 📁 reports/                       # Evaluation reports
│   ├── logistic_regression_test_report.json
│   └── random_forest_test_report.json
│
├── 📁 dashboard/                     # Streamlit web interface
│   ├── app.py                        # Main dashboard entry point
│   ├── pages/                        # Dashboard pages
│   │   ├── 1_🏠_Home.py              # Home page with overview
│   │   ├── 2_🔮_Predictor.py         # Single project prediction
│   │   ├── 3_📈_Analytics.py         # Dataset analytics
│   │   ├── 4_⚖️_Model_Performance.py # Model evaluation metrics
│   │   └── 5_📤_Data_Upload.py       # Upload custom data for batch analysis
│   └── components/                   # Reusable UI components
│
├── 📁 code/                          # Source code modules
│   ├── data_prep/                    # Data preprocessing modules
│   │   ├── load_data.py             # Load raw/processed data
│   │   ├── validate_data.py         # Data validation
│   │   ├── clean_data.py            # Data cleaning
│   │   ├── encode_features.py       # Feature encoding
│   │   ├── split_data.py            # Train/val/test split
│   │   └── select_features.py       # Feature selection (L1 diagnostic)
│   │
│   ├── models/                       # Model training modules
│   │   ├── train_logreg.py          # Logistic Regression training
│   │   ├── train_rf.py              # Random Forest training
│   │   ├── calibrate.py             # Model calibration
│   │   ├── evaluate.py              # Model evaluation
│   │   ├── ordinal_metrics.py       # Ordinal evaluation metrics
│   │   └── model_registry.py        # Model version management
│   │
│   ├── explainability/               # Model explanation modules
│   │   ├── shap_engine.py           # SHAP-based explanations
│   │   └── narrative_builder.py    # Natural language explanations
│   │
│   ├── api/                          # Flask REST API
│   │   ├── app.py                   # Flask application factory
│   │   ├── routes.py                # API endpoints
│   │   ├── schemas.py               # Pydantic schemas
│   │   ├── inference.py             # Prediction inference
│   │   └── errors.py                # Error handlers
│   │
│   └── utils/                        # Utility functions
│       ├── config.py                # Configuration & feature specs
│       └── risk_levels.py           # Risk level definitions
│
├── 📁 tests/                         # Test modules
│   ├── test_data_prep.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_explainability.py
│
└── 📁 deployment/                    # Docker deployment files
    ├── Dockerfile
    └── docker-compose.yml
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training pipeline:**
   ```bash
   python pipeline.py
   ```
   This will:
   - Load and clean the raw dataset
   - Split into train/val/test sets
   - Train Logistic Regression and Random Forest models
   - Evaluate models and save reports

3. **Launch the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

4. **Start the API server:**
   ```bash
   python -m src.api.app
   ```

## 📊 Key Files Explained

- **pipeline.py**: The main entry point for training models. Run this first to generate models and processed data.
- **src/utils/config.py**: Single source of truth for feature specifications and paths.
- **src/utils/risk_levels.py**: Defines the 4 risk levels (Low, Medium, High, Critical) with colors and actions.
- **dashboard/app.py**: Streamlit dashboard with multiple pages for prediction, analytics, and model performance.

## 🔧 Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `data_prep/` | Loading, validating, cleaning, encoding, and splitting data |
| `models/` | Training, calibrating, and evaluating models |
| `explainability/` | Generating SHAP explanations and natural language narratives |
| `api/` | REST API for model inference and explanations |
| `utils/` | Configuration, constants, and shared utilities |

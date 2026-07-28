# Project Risk Decision Support System (DSS)

A comprehensive machine learning system for predicting project risk levels with explainable AI. The system uses a four-level ordinal classification (Low, Medium, High, Critical) and provides SHAP-based explanations for predictions.

## 🎯 Project Overview

This DSS helps project managers and stakeholders assess project risk by analyzing 18 key features across project characteristics, team composition, and environmental factors. The system includes:

- **Data Pipeline**: Automated data loading, cleaning, validation, and preprocessing
- **Model Training**: Logistic Regression and Random Forest models with ordinal evaluation
- **Explainability**: SHAP-based local and global explanations with natural language narratives
- **REST API**: Flask API for programmatic access to predictions
- **Interactive Dashboard**: Streamlit web interface for risk prediction and analytics

## 📊 Dataset

- **Source**: Project Management Risk Raw Dataset (synthetic, 4000 records)
- **Target**: Risk_Level (4 ordinal levels: Low, Medium, High, Critical)
- **Features**: 18 features including project type, complexity score, methodology, team experience, resource availability, etc.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
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
- Train Logistic Regression and Random Forest models
- Evaluate models and save reports

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
│   └── model_card.json               # Model metadata
│
├── 📁 reports/                       # Evaluation reports
│
├── 📁 dashboard/                     # Streamlit web interface
│   ├── app.py                        # Main dashboard entry
│   ├── pages/                        # Dashboard pages
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
└── 📁 deployment/                    # Docker deployment files
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
- **SHAP Values**: Feature importance for individual predictions
- **Natural Language Narratives**: Plain-English explanations
- **Waterfall Plots**: Visual explanation of feature contributions

## 📈 Model Performance

After running the pipeline, you'll see evaluation metrics:

```
Logistic Regression:
  Accuracy: 0.5017, QWK: 0.6657, Within-one: 0.9050

Random Forest:
  Accuracy: 0.5067, QWK: 0.6219, Within-one: 0.9267
```

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
    "Low": 0.1,
    "Medium": 0.3,
    "High": 0.4,
    "Critical": 0.2
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
pytest tests/
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 📚 Module Documentation

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
- `calibrate.py`: Model calibration for better probabilities
- `evaluate.py`: Model evaluation with ordinal metrics
- `ordinal_metrics.py`: QWK, within-one accuracy, macro F1
- `model_registry.py`: Model version management

### Explainability (`code/explainability/`)
- `shap_engine.py`: SHAP value computation for tree and linear models
- `narrative_builder.py`: Natural language explanation generation

### API (`code/api/`)
- `app.py`: Flask application factory
- `routes.py`: API endpoints (/predict, /explain, /health)
- `schemas.py`: Pydantic request/response schemas
- `inference.py`: Prediction inference logic
- `errors.py`: Error handling and validation

### Dashboard (`dashboard/`)
- `app.py`: Main Streamlit application
- `pages/`: Home, Predictor, Analytics, Model Performance, Data Upload
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

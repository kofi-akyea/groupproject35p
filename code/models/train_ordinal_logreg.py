"""
Ordinal Logistic Regression training module.
Uses mord library for ordinal regression.
"""
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import mord
from code.utils.config import FEATURE_NAMES, FEATURE_SPECS, MODELS_DIR
from code.utils.risk_levels import RISK_LEVELS

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_ordinal_logreg_pipeline():
    """
    Build an ordinal logistic regression pipeline.
    
    Mathematical Foundation:
    Ordinal Logistic Regression models the cumulative probabilities of the ordered classes.
    
    For K ordered classes (1, 2, ..., K), the model estimates:
    
    P(Y ≤ k | X) = σ(θ_k - β·X)
    
    Where:
    - σ is the logistic sigmoid function: σ(z) = 1 / (1 + e^(-z))
    - θ_k are threshold parameters for each class boundary
    - β are the feature coefficients
    - X is the feature vector
    
    The probability of each class is:
    P(Y = k | X) = P(Y ≤ k | X) - P(Y ≤ k-1 | X)
    
    This model explicitly accounts for the ordinal nature of the target variable.
    """
    # Categorize input feature specifications into categorical and numeric subsets
    categorical_features = [f.name for f in FEATURE_SPECS if f.kind == "nominal" or f.kind == "ordinal"]
    numeric_features = [f.name for f in FEATURE_SPECS if f.kind == "numeric"]
    
    # Define explicit category orders for ordinal features
    ordinal_categories = {
        "Project_Phase": ["Initiation", "Planning", "Execution", "Monitoring", "Closure"],
        "Team_Experience_Level": ["Junior", "Mixed", "Senior", "Expert"],
        "Project_Manager_Experience": ["Junior PM", "Mid-level PM", "Senior PM", "Certified PM"],
        "Requirement_Stability": ["Volatile", "Moderate", "Stable"],
        "Risk_Management_Maturity": ["None", "Basic", "Formal", "Advanced"],
        "Change_Control_Maturity": ["None", "Basic", "Formal", "Advanced"],
        "Tech_Environment_Stability": ["Legacy/Unstable", "Mixed", "Modern/Stable"]
    }
    
    # Configure numeric preprocessor step: median imputation + standard scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    nominal_features = [f for f in categorical_features if f not in ordinal_categories]
    ordinal_feature_list = [f for f in categorical_features if f in ordinal_categories]
    
    # Configure nominal preprocessor step: mode imputation + one-hot encoding
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Configure ordinal preprocessor step: mode imputation + ordinal integer encoding
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[ordinal_categories[f] for f in ordinal_feature_list], handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Combine feature preprocessing transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_feature_list)
        ],
        remainder='drop'
    )
    
    # Construct complete pipeline with mord.LogisticAT (All-Threshold) ordinal model
    pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', mord.LogisticAT(alpha=1.0))
    ])
    
    return pipeline


def train_ordinal_logreg(X_train, y_train):
    """
    Train ordinal logistic regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or np.array
        Training labels (0, 1, 2, 3 for Low, Medium, High, Critical)
    
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline
    """
    # Fit ordinal logistic regression pipeline on training set
    pipeline = build_ordinal_logreg_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Save model artifact to models/ordinal_logistic_regression.joblib
    joblib.dump(pipeline, MODELS_DIR / "ordinal_logistic_regression.joblib")
    
    print("Ordinal Logistic Regression model saved to: models/ordinal_logistic_regression.joblib")
    
    return pipeline


if __name__ == "__main__":
    # Standalone execution testing block
    import pandas as pd
    from code.data_prep.load_data import load_raw
    from code.data_prep.clean_data import clean
    from code.data_prep.encode_features import encode_target
    from code.data_prep.split_data import split
    
    print("Loading and preparing data...")
    df = clean(load_raw())
    X = df[FEATURE_NAMES].copy()
    y = encode_target(df["Risk_Level"])
    
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
    
    print("Training Ordinal Logistic Regression...")
    pipeline = train_ordinal_logreg(X_train, y_train)
    
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

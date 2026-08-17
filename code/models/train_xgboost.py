"""
XGBoost training module for multiclass classification.
"""
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import xgboost as xgb
from code.utils.config import FEATURE_NAMES, FEATURE_SPECS, MODELS_DIR
from code.utils.risk_levels import RISK_LEVELS

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_xgboost_pipeline():
    """
    Build an XGBoost pipeline for multiclass classification.
    
    Mathematical Foundation:
    XGBoost (Extreme Gradient Boosting) is an ensemble of decision trees trained using gradient boosting.
    
    For a multiclass problem with K classes, XGBoost minimizes the softmax objective:
    
    Obj(θ) = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
    
    Where:
    - L is the loss function (multi:softprob for multiclass)
    - ŷ_i = softmax(Σ f_k(x_i)) where f_k are the tree functions
    - Ω is the regularization term: Ω(f) = γT + 0.5λ||w||^2
    - T is the number of leaves, w is the leaf weights
    - γ and λ are regularization parameters
    
    The softmax function converts scores to probabilities:
    
    P(Y = k | X) = exp(z_k) / Σ_j exp(z_j)
    
    Where z_k is the score for class k.
    
    The algorithm builds trees sequentially, each tree correcting the errors of the previous ones.
    """
    # Separate categorical and numeric features
    categorical_features = [f.name for f in FEATURE_SPECS if f.kind == "nominal" or f.kind == "ordinal"]
    numeric_features = [f.name for f in FEATURE_SPECS if f.kind == "numeric"]
    
    # Define ordinal categories for ordinal features
    ordinal_categories = {
        "Project_Phase": ["Initiation", "Planning", "Execution", "Monitoring", "Closure"],
        "Team_Experience_Level": ["Junior", "Mixed", "Senior", "Expert"],
        "Project_Manager_Experience": ["Junior PM", "Mid-level PM", "Senior PM", "Certified PM"],
        "Requirement_Stability": ["Volatile", "Moderate", "Stable"],
        "Risk_Management_Maturity": ["None", "Basic", "Formal", "Advanced"],
        "Change_Control_Maturity": ["None", "Basic", "Formal", "Advanced"],
        "Tech_Environment_Stability": ["Legacy/Unstable", "Mixed", "Modern/Stable"]
    }
    
    # Build preprocessor with median imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    nominal_features = [f for f in categorical_features if f not in ordinal_categories]
    ordinal_feature_list = [f for f in categorical_features if f in ordinal_categories]
    
    # Configure nominal and ordinal feature encoders
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[ordinal_categories[f] for f in ordinal_feature_list], handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Combine feature preprocessing steps into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_feature_list)
        ],
        remainder='drop'
    )
    
    # Build complete pipeline with XGBoost Classifier
    pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=4,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline


def train_xgboost(X_train, y_train):
    """
    Train XGBoost model.
    
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
    # Instantiate and fit XGBoost model pipeline
    pipeline = build_xgboost_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Save trained model pipeline to models/xgboost.joblib
    joblib.dump(pipeline, MODELS_DIR / "xgboost.joblib")
    
    print("XGBoost model saved to: models/xgboost.joblib")
    
    return pipeline


if __name__ == "__main__":
    # Standalone test script execution
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
    
    print("Training XGBoost...")
    pipeline = train_xgboost(X_train, y_train)
    
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

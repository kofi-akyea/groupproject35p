"""
SVM (RBF kernel) training module for multiclass classification.
"""
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from code.utils.config import FEATURE_NAMES, FEATURE_SPECS, MODELS_DIR
from code.utils.risk_levels import RISK_LEVELS

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_svm_pipeline():
    """
    Build an SVM pipeline with RBF kernel for multiclass classification.
    
    Mathematical Foundation:
    Support Vector Machine with RBF kernel finds the optimal hyperplane that separates classes.
    
    For binary classification, the decision function is:
    
    f(x) = Σ α_i y_i K(x_i, x) + b
    
    Where:
    - α_i are Lagrange multipliers (learned during training)
    - y_i are the training labels (-1 or +1 for binary)
    - K(x_i, x) is the kernel function
    - b is the bias term
    
    The RBF (Radial Basis Function) kernel is:
    
    K(x_i, x) = exp(-γ ||x_i - x||^2)
    
    Where:
    - γ is the kernel coefficient (gamma)
    - ||x_i - x||^2 is the squared Euclidean distance
    
    For multiclass classification (one-vs-one strategy):
    - K*(K-1)/2 binary classifiers are trained
    - Each classifier separates one pair of classes
    - Voting determines the final class
    
    The probability estimation uses Platt scaling:
    
    P(Y = k | X) = 1 / (1 + exp(A_k f_k(X) + B_k))
    
    Where A_k and B_k are learned parameters for class k.
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
    
    # Configure numeric preprocessor step (imputation + scaling)
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
    
    # Combine feature preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_feature_list)
        ],
        remainder='drop'
    )
    
    # Build complete pipeline with SVM (RBF kernel)
    pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            decision_function_shape='ovr',
            random_state=42
        ))
    ])
    
    return pipeline


def train_svm(X_train, y_train):
    """
    Train SVM model with RBF kernel.
    
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
    # Instantiate and fit SVM model pipeline
    pipeline = build_svm_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Save model artifact to models/svm_rbf.joblib
    joblib.dump(pipeline, MODELS_DIR / "svm_rbf.joblib")
    
    print("SVM (RBF kernel) model saved to: models/svm_rbf.joblib")
    
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
    
    print("Training SVM (RBF kernel)...")
    pipeline = train_svm(X_train, y_train)
    
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

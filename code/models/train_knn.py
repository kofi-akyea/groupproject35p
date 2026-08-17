"""
K-Nearest Neighbors (KNN) training module for multiclass classification.
"""
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from code.utils.config import FEATURE_NAMES, FEATURE_SPECS, MODELS_DIR
from code.utils.risk_levels import RISK_LEVELS

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_knn_pipeline():
    """
    Build a KNN pipeline for multiclass classification.
    
    Mathematical Foundation:
    K-Nearest Neighbors is a non-parametric, instance-based learning algorithm.
    It classifies a new instance based on the majority class among its k nearest neighbors.
    
    For a test instance x, the prediction is:
    
    ŷ = argmax_k Σ_{i ∈ N_k(x)} I(y_i = k)
    
    Where:
    - N_k(x) is the set of k nearest neighbors to x
    - I(y_i = k) is an indicator function (1 if y_i = k, else 0)
    - Distance is measured using Minkowski distance (p=2 for Euclidean)
    
    The distance metric is:
    
    d(x_i, x_j) = (Σ |x_i - x_j|^p)^(1/p)
    
    For Euclidean distance (p=2):
    
    d(x_i, x_j) = sqrt(Σ (x_i - x_j)^2)
    
    Distance-based weighting gives more importance to closer neighbors:
    
    w_i = 1 / d(x_i, x)
    
    The weighted vote for class k is:
    
    vote_k = Σ_{i ∈ N_k(x)} w_i * I(y_i = k)
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
    
    # Build numeric preprocessor step (imputation + scaling)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    nominal_features = [f for f in categorical_features if f not in ordinal_categories]
    ordinal_feature_list = [f for f in categorical_features if f in ordinal_categories]
    
    # Configure nominal and ordinal encoders
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
    
    # Build complete pipeline with KNeighborsClassifier
    pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2,
            n_jobs=1
        ))
    ])
    
    return pipeline


def train_knn(X_train, y_train):
    """
    Train KNN model.
    
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
    # Instantiate and fit KNN pipeline
    pipeline = build_knn_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Save model artifact to models/knn.joblib
    joblib.dump(pipeline, MODELS_DIR / "knn.joblib")
    
    print("KNN model saved to: models/knn.joblib")
    
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
    
    print("Training KNN...")
    pipeline = train_knn(X_train, y_train)
    
    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

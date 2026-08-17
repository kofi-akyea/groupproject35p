"""API routes for prediction and explanation."""
from flask import Blueprint, request, jsonify
from code.api.schemas import ProjectFeatures
from code.api.inference import predict
from code.api.errors import error_response, validate_payload

# Initialize Flask blueprint for API endpoints
bp = Blueprint("api", __name__)


@bp.route("/predict", methods=["POST"])
def route_predict():
    """Predict project risk level from input features."""
    # Parse incoming JSON payload safely
    payload = request.get_json(silent=True) or {}
    
    # Validate payload parameters against schema rules
    err = validate_payload(payload)
    if err:
        return error_response(err, 400)
    
    # Map payload into ProjectFeatures dataclass and run model inference
    features = ProjectFeatures(**payload)
    result = predict(features, model_name=payload.get("model_name", "random_forest"))
    
    # Return JSON response containing prediction, probabilities, SHAP values, and narrative
    return jsonify({
        "request_id": result.request_id,
        "prediction": result.prediction,
        "probabilities": result.probabilities,
        "shap": result.shap,
        "top_features": result.top_features,
        "narrative": result.narrative,
        "model_version": result.model_version,
    })


@bp.route("/explain", methods=["POST"])
def route_explain():
    """Return explanation-capable prediction output for supported tree-based models."""
    # Alias endpoint for explanation requests
    return route_predict()


@bp.route("/health", methods=["GET"])
def route_health():
    """Health check endpoint."""
    import os
    # Verify trained model joblib files exist on disk
    required_models = [
        "models/logistic_regression.joblib",
        "models/ordinal_logistic_regression.joblib",
        "models/random_forest.joblib",
        "models/xgboost.joblib",
        "models/svm_rbf.joblib",
        "models/knn.joblib",
    ]
    models_exist = all(os.path.exists(path) for path in required_models)
    
    # Return status payload
    return jsonify({
        "status": "ok",
        "models_loaded": models_exist,
        "version": "1.0.0",
    })
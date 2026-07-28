"""API routes for prediction and explanation."""
from flask import Blueprint, request, jsonify
from code.api.schemas import ProjectFeatures
from code.api.inference import predict
from code.api.errors import error_response, validate_payload

bp = Blueprint("api", __name__)


@bp.route("/predict", methods=["POST"])
def route_predict():
    """Predict project risk level from input features."""
    payload = request.get_json(silent=True) or {}
    err = validate_payload(payload)
    if err:
        return error_response(err, 400)
    features = ProjectFeatures(**payload)
    result = predict(features, model_name=payload.get("model", "random_forest"))
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
    """Alias for /predict - reserved for explainability-only clients."""
    return route_predict()


@bp.route("/health", methods=["GET"])
def route_health():
    """Health check endpoint."""
    import os
    models_exist = all(
        os.path.exists(f)
        for f in ["models/random_forest.joblib", "models/logistic_regression.joblib"]
    )
    return jsonify({
        "status": "ok",
        "models_loaded": models_exist,
        "version": "1.0.0",
    })
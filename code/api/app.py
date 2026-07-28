"""Flask application factory for the Project Risk DSS API."""
from flask import Flask
from code.api.routes import bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(bp)
    return app


# WSGI entrypoint
app = create_app()
from flask import Flask
from config.database import init_db
from app.routes import train_bp, status_bp, list_models_bp, predict_bp, test_server_bp

def create_app():
    app = Flask(__name__)

    # Initialize database
    init_db()

    # Register blueprints
    app.register_blueprint(test_server_bp)
    app.register_blueprint(train_bp)
    app.register_blueprint(status_bp)
    app.register_blueprint(list_models_bp)
    app.register_blueprint(predict_bp)

    return app
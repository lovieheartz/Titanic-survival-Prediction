# app/__init__.py

# Initializes the Flask app using the factory pattern.

from flask import Flask

def create_app():
    app = Flask(__name__)
    from .routes import main
    app.register_blueprint(main)
    return app

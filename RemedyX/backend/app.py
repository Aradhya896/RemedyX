"""
app.py
------
The main Flask application for RemedyX.

How Flask works (simple explanation):
- Flask is a Python library that listens for HTTP requests.
- You define "routes" (URL paths) and what function to call for each.
- When the browser sends a request to /search, Flask calls the search() function.
- The function processes the request and returns a JSON response.

Startup sequence:
1. Load environment config from .env
2. Load BGE-M3 model + FAISS index + BM25 index (ONCE — stored in app.config)
3. Register all route Blueprints
4. Start the Flask server

The indexes/model are stored in app.config["RETRIEVAL_COMPONENTS"].
Every route handler can access them via current_app.config["RETRIEVAL_COMPONENTS"].
This is how we avoid reloading the model on every request.
"""

import os
import sys

# Add the project root to Python's path so we can import from src/
# This file is at: backend/app.py
# Project root is one level up
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, jsonify
from flask_cors import CORS

from backend.config import Config
from backend.routes.search_routes import search_bp
from backend.routes.ticket_routes import ticket_bp
from backend.routes.feedback_routes import feedback_bp


def create_app():
    """
    Application Factory function.

    Why use a factory function instead of just creating app = Flask(__name__)?
    - It makes testing easier (you can create multiple app instances in tests).
    - It's the recommended Flask pattern for structured projects.
    - You can pass different configs for development vs. production.

    Returns:
        A configured Flask app instance.
    """

    app = Flask(__name__)

    # -----------------------------------------
    # 1. Configure Flask
    # -----------------------------------------

    app.config["SECRET_KEY"]  = Config.SECRET_KEY
    app.config["TOP_K"]       = Config.TOP_K
    app.config["CANDIDATE_K"] = Config.CANDIDATE_K
    app.config["HAS_LLM"]     = Config.has_llm()
    app.config["HAS_DATABASE"] = Config.has_database()
    app.config["GEMINI_API_KEY"] = Config.GEMINI_API_KEY

    # -----------------------------------------
    # 2. Enable CORS
    # -----------------------------------------

    # CORS = Cross-Origin Resource Sharing
    # By default, browsers block JavaScript from calling APIs on different ports.
    # Our frontend runs on port 5500 (VS Code Live Server) or opened as a file,
    # while our Flask backend runs on port 5000.
    # Flask-CORS tells the browser: "Yes, it's okay to talk to this server."
    CORS(app, resources={r"/*": {"origins": "*"}})

    # -----------------------------------------
    # 3. Load retrieval components ONCE at startup
    # -----------------------------------------

    print("\n[App] Loading retrieval components...")
    print("[App] This takes ~10-20 seconds for BGE-M3 to load.")

    try:
        from src.retrieval.loader import load_all
        components = load_all()
        app.config["RETRIEVAL_COMPONENTS"] = components
        print(f"[App] Retrieval components loaded. Tickets: {len(components['ticket_df'])}")
    except Exception as e:
        print(f"[App] CRITICAL: Failed to load retrieval components: {e}")
        print("[App] Search endpoints will return errors until this is fixed.")
        app.config["RETRIEVAL_COMPONENTS"] = None

    # -----------------------------------------
    # 4. Register route Blueprints
    # -----------------------------------------

    # A Blueprint groups related routes.
    # We pass url_prefix so all routes in search_bp start with nothing (root),
    # making endpoints like /health, /search, /resolve, /tickets, /feedback
    app.register_blueprint(search_bp)
    app.register_blueprint(ticket_bp)
    app.register_blueprint(feedback_bp)

    # -----------------------------------------
    # 5. Global error handlers
    # -----------------------------------------

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error."}), 500

    print("\n[App] Flask app created successfully.")
    print("[App] Available endpoints:")
    print("       GET  /health")
    print("       POST /search")
    print("       POST /resolve")
    print("       POST /tickets")
    print("       GET  /tickets")
    print("       GET  /tickets/stats")
    print("       POST /feedback")
    print("       GET  /feedback")

    return app


# -----------------------------------------
# Entry point: run this file directly to start the server
# python backend/app.py
# -----------------------------------------

if __name__ == "__main__":
    app = create_app()

    port = Config.FLASK_PORT
    debug = Config.FLASK_ENV == "development"

    print(f"\n[App] Starting server on http://localhost:{port}")
    print(f"[App] Debug mode: {debug}")
    print("[App] Press CTRL+C to stop.\n")

    app.run(
        host="0.0.0.0",   # Accept connections from any network interface
        port=port,
        debug=debug,
        use_reloader=False  # Disable reloader — it would reload the ML model twice!
    )

"""
feedback_routes.py
------------------
Flask routes for collecting user feedback on search results.

Endpoints:
  POST /feedback   — Submit feedback about a search result
  GET  /feedback   — Get all feedback entries (optional admin view)
"""

from flask import Blueprint, request, jsonify, current_app

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/feedback", methods=["POST"])
def submit_feedback():
    """
    POST /feedback
    --------------
    Saves user feedback about search/resolution quality to MySQL.

    Request body (JSON):
        {
            "query":         "user account locked",
            "rating":        4,
            "feedback_text": "Results were very helpful, resolved quickly."
        }

    Rules:
        - rating must be 1 to 5
        - feedback_text is optional
        - query is optional (but helpful for analysis)

    Response:
        201 Created — { "message": "Feedback submitted. Thank you!" }
        400 Bad Request — validation error
        503 — database not configured
    """

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided."}), 400

    # Validate rating (required, must be integer 1-5)
    if "rating" not in data:
        return jsonify({"error": "Field 'rating' is required."}), 400

    try:
        rating = int(data["rating"])
    except (ValueError, TypeError):
        return jsonify({"error": "Rating must be a number."}), 400

    if rating < 1 or rating > 5:
        return jsonify({"error": "Rating must be between 1 and 5."}), 400

    # Optional fields
    query         = str(data.get("query", "")).strip()[:500]
    feedback_text = str(data.get("feedback_text", "")).strip()[:1000]

    if not current_app.config.get("HAS_DATABASE"):
        return jsonify({
            "error": "Database not configured. Please set DB_PASSWORD in .env file."
        }), 503

    try:
        from src.database.db import insert_feedback
        insert_feedback(
            query=query,
            rating=rating,
            feedback_text=feedback_text
        )
        return jsonify({
            "message": "Feedback submitted. Thank you!"
        }), 201

    except Exception as e:
        current_app.logger.error(f"Failed to insert feedback: {e}")
        return jsonify({"error": "Failed to save feedback."}), 500


@feedback_bp.route("/feedback", methods=["GET"])
def get_feedback():
    """
    GET /feedback
    -------------
    Returns all feedback entries from MySQL (most recent first).
    """

    if not current_app.config.get("HAS_DATABASE"):
        return jsonify({"feedback": [], "total": 0, "note": "Database not configured."}), 200

    try:
        from src.database.db import get_all_feedback
        feedback = get_all_feedback()
        return jsonify({"feedback": feedback, "total": len(feedback)}), 200

    except Exception as e:
        current_app.logger.error(f"Failed to fetch feedback: {e}")
        return jsonify({"error": "Failed to fetch feedback."}), 500

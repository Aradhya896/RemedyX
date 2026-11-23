"""
ticket_routes.py
----------------
Flask routes for submitting and retrieving new support tickets.

Endpoints:
  POST /tickets          — Submit a new ticket (saves to MySQL)
  GET  /tickets          — Get all submitted tickets from MySQL
  GET  /tickets/stats    — Get summary statistics for dashboard
"""

from flask import Blueprint, request, jsonify, current_app

ticket_bp = Blueprint("tickets", __name__)


@ticket_bp.route("/tickets", methods=["POST"])
def submit_ticket():
    """
    POST /tickets
    -------------
    Saves a newly submitted support ticket to MySQL.

    Request body (JSON):
        {
            "title":       "Cannot login to system",
            "description": "My account is locked and I cannot access...",
            "category":    "Account Management",
            "priority":    "High"
        }

    Response:
        201 Created — { "message": "Ticket submitted.", "ticket_id": 42 }
        400 Bad Request — if required fields are missing
        503 Service Unavailable — if database is not configured
    """

    data = request.get_json()

    # Validate required fields
    if not data:
        return jsonify({"error": "No data provided."}), 400

    required_fields = ["title", "description", "category", "priority"]
    for field in required_fields:
        if field not in data or not str(data[field]).strip():
            return jsonify({"error": f"Field '{field}' is required."}), 400

    # Check if database is available
    if not current_app.config.get("HAS_DATABASE"):
        return jsonify({
            "error": "Database not configured. Please set DB_PASSWORD in .env file."
        }), 503

    # Import database functions
    try:
        from src.database.db import insert_ticket
        ticket_id = insert_ticket(
            title=data["title"].strip(),
            description=data["description"].strip(),
            category=data["category"].strip(),
            priority=data["priority"].strip()
        )
        return jsonify({
            "message": "Ticket submitted successfully.",
            "ticket_id": ticket_id
        }), 201

    except Exception as e:
        current_app.logger.error(f"Failed to insert ticket: {e}")
        return jsonify({"error": "Failed to save ticket. Please try again."}), 500


@ticket_bp.route("/tickets", methods=["GET"])
def get_tickets():
    """
    GET /tickets
    ------------
    Returns all submitted tickets from MySQL (most recent first).

    Response:
        200 OK — { "tickets": [...], "total": N }
    """

    if not current_app.config.get("HAS_DATABASE"):
        return jsonify({"tickets": [], "total": 0, "note": "Database not configured."}), 200

    try:
        from src.database.db import get_all_tickets
        tickets = get_all_tickets()
        return jsonify({"tickets": tickets, "total": len(tickets)}), 200

    except Exception as e:
        current_app.logger.error(f"Failed to fetch tickets: {e}")
        return jsonify({"error": "Failed to fetch tickets."}), 500


@ticket_bp.route("/tickets/stats", methods=["GET"])
def get_stats():
    """
    GET /tickets/stats
    ------------------
    Returns summary stats used by the Dashboard page.

    Response:
        {
            "total_historical":  5785,
            "total_submitted":   12,
            "total_feedback":    5,
            "total_searches":    34
        }
    """

    # Historical tickets come from the FAISS index (always available)
    components = current_app.config.get("RETRIEVAL_COMPONENTS", {})
    ticket_df  = components.get("ticket_df")
    total_historical = len(ticket_df) if ticket_df is not None else 0

    # Submitted tickets, feedback, searches come from MySQL
    total_submitted = 0
    total_feedback  = 0
    total_searches  = 0

    if current_app.config.get("HAS_DATABASE"):
        try:
            from src.database.db import get_counts
            counts = get_counts()
            total_submitted = counts.get("tickets", 0)
            total_feedback  = counts.get("feedback", 0)
            total_searches  = counts.get("searches", 0)
        except Exception as e:
            current_app.logger.warning(f"Could not fetch DB counts: {e}")

    return jsonify({
        "total_historical": total_historical,
        "total_submitted":  total_submitted,
        "total_feedback":   total_feedback,
        "total_searches":   total_searches,
    }), 200

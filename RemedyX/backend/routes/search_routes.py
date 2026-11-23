"""
search_routes.py
----------------
Flask routes (endpoints) related to ticket searching and resolution.

Endpoints defined here:
  GET  /health   — Check if the backend is running
  POST /search   — Search for similar historical tickets
  POST /resolve  — Get resolution suggestion (with or without LLM)

How Flask routes work:
  @search_bp.route("/search", methods=["POST"])
  def search():
      ...
  This tells Flask: "when a POST request comes in at /search, call search()"

We use a Blueprint here (search_bp).
A Blueprint is just a way to group related routes in a separate file,
so app.py doesn't become one massive file.
"""

from flask import Blueprint, request, jsonify
from src.retrieval.hybrid_search import hybrid_search

# Create a Blueprint named "search"
# The frontend will send requests to /search, /resolve, /health
search_bp = Blueprint("search", __name__)


@search_bp.route("/health", methods=["GET"])
def health():
    """
    GET /health
    -----------
    Simple check to see if the backend is alive.
    The frontend can call this to show a "connected" indicator.

    Returns:
        200 OK with { "status": "ok", "message": "..." }
    """
    return jsonify({
        "status": "ok",
        "message": "RemedyX backend is running."
    }), 200


@search_bp.route("/search", methods=["POST"])
def search():
    """
    POST /search
    ------------
    Accepts an IT problem query and returns similar historical tickets.

    Request body (JSON):
        { "query": "user cannot login, account locked" }

    Response (JSON):
        {
            "query": "user cannot login, account locked",
            "results": [
                {
                    "id": "430197",
                    "title": null,
                    "description": "...",
                    "resolution": "We have unlocked the account...",
                    "classification": "...",
                    "priority": "S4_DCO",
                    "status": "Closed",
                    "score": 0.032266
                },
                ...
            ],
            "total": 5
        }

    Error responses:
        400 — if query is missing or empty
        500 — if retrieval fails
    """

    # Step 1: Parse incoming JSON body
    data = request.get_json()

    # Step 2: Validate — make sure query is present and not empty
    if not data or "query" not in data:
        return jsonify({
            "error": "Missing 'query' field in request body."
        }), 400

    query = data["query"].strip()

    if not query:
        return jsonify({
            "error": "Query cannot be empty."
        }), 400

    if len(query) > 1000:
        return jsonify({
            "error": "Query is too long. Maximum 1000 characters."
        }), 400

    # Step 3: Get retrieval components from app context
    # (These were loaded once at Flask startup — see app.py)
    # current_app.config holds the loaded indexes and model
    from flask import current_app
    components = current_app.config["RETRIEVAL_COMPONENTS"]
    top_k      = current_app.config["TOP_K"]
    candidate_k = current_app.config["CANDIDATE_K"]

    # Step 4: Run hybrid retrieval
    try:
        results = hybrid_search(
            query=query,
            faiss_index=components["faiss_index"],
            bm25_index=components["bm25_index"],
            model=components["model"],
            ticket_df=components["ticket_df"],
            top_k=top_k,
            candidate_k=candidate_k,
        )
    except Exception as e:
        return jsonify({
            "error": f"Retrieval failed: {str(e)}"
        }), 500

    # Step 5: Return results as JSON
    return jsonify({
        "query":   query,
        "results": results,
        "total":   len(results)
    }), 200


@search_bp.route("/resolve", methods=["POST"])
def resolve():
    """
    POST /resolve
    -------------
    Similar to /search but also returns a generated resolution suggestion.
    If no LLM API key is configured, falls back to showing the best
    resolution from retrieved tickets.

    Request body (JSON):
        { "query": "user cannot login, account locked" }

    Response (JSON):
        {
            "query": "...",
            "results": [...],
            "resolution": {
                "text": "Based on similar tickets, the resolution is...",
                "source": "llm"   OR   "retrieved"   OR   "none"
            }
        }
    """

    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field."}), 400

    query = data["query"].strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    from flask import current_app
    components = current_app.config["RETRIEVAL_COMPONENTS"]
    top_k      = current_app.config["TOP_K"]
    candidate_k = current_app.config["CANDIDATE_K"]
    has_llm    = current_app.config["HAS_LLM"]

    # Step 1: Retrieve similar tickets
    try:
        results = hybrid_search(
            query=query,
            faiss_index=components["faiss_index"],
            bm25_index=components["bm25_index"],
            model=components["model"],
            ticket_df=components["ticket_df"],
            top_k=top_k,
            candidate_k=candidate_k,
        )
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # Step 2: Generate resolution
    resolution = build_resolution(query, results, has_llm, current_app)

    return jsonify({
        "query":      query,
        "results":    results,
        "resolution": resolution,
        "total":      len(results)
    }), 200


def build_resolution(query, results, has_llm, app):
    """
    Build a resolution suggestion based on retrieved tickets.

    If LLM is available: use it to generate a natural language answer.
    If not: pick the best available resolution from retrieved tickets.

    Returns:
        A dict with keys: "text" (string) and "source" ("llm", "retrieved", or "none")
    """

    # Try LLM first if configured
    if has_llm:
        try:
            from src.rag.answer_generator import generate_answer
            answer = generate_answer(query, results, app.config["GEMINI_API_KEY"])
            return {"text": answer, "source": "llm"}
        except Exception as e:
            # LLM failed — fall through to retrieved resolution
            app.logger.warning(f"LLM generation failed: {e}. Falling back.")

    # Fallback: find the best resolution from retrieved tickets
    for ticket in results:
        resolution = ticket.get("resolution")
        if resolution:
            return {
                "text": resolution,
                "source": "retrieved",
                "source_ticket_id": ticket.get("id")
            }

    # No resolution found in any retrieved ticket
    return {
        "text": "No resolution found in similar historical tickets. Please escalate manually.",
        "source": "none"
    }

"""
db.py
-----
MySQL database connection and query functions for RemedyX.

NOTE: This is a STUB for Phase 3.
      Full MySQL implementation comes in Phase 4.
      These placeholder functions allow Flask to start without crashing
      even before the database is set up.

In Phase 4 we will implement:
  - get_connection()     — returns a MySQL connection
  - insert_ticket()      — saves a new ticket
  - get_all_tickets()    — retrieves all submitted tickets
  - insert_feedback()    — saves user feedback
  - get_all_feedback()   — retrieves all feedback
  - get_counts()         — gets record counts for dashboard stats
"""


def get_connection():
    """Returns a MySQL database connection. Implemented in Phase 4."""
    raise NotImplementedError("Database not configured yet. Coming in Phase 4.")


def insert_ticket(title, description, category, priority):
    """Saves a new support ticket to MySQL. Implemented in Phase 4."""
    raise NotImplementedError("Database not configured yet. Coming in Phase 4.")


def get_all_tickets():
    """Returns all submitted tickets from MySQL. Implemented in Phase 4."""
    return []


def insert_feedback(query, rating, feedback_text):
    """Saves user feedback to MySQL. Implemented in Phase 4."""
    raise NotImplementedError("Database not configured yet. Coming in Phase 4.")


def get_all_feedback():
    """Returns all feedback entries from MySQL. Implemented in Phase 4."""
    return []


def get_counts():
    """Returns record counts for dashboard stats. Implemented in Phase 4."""
    return {"tickets": 0, "feedback": 0, "searches": 0}

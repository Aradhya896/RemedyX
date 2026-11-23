"""
config.py
---------
Reads all configuration from the .env file and exposes them
as simple Python variables.

Why do we use a .env file?
- Keeps secrets (passwords, API keys) out of source code.
- Different developers can have different settings without
  changing any Python files.
- When you deploy the project, you just change the .env file.

How to use:
  from backend.config import Config
  print(Config.DB_HOST)
"""

import os
from dotenv import load_dotenv

# Load the .env file from the project root
# dotenv finds the .env file and sets each line as an environment variable
# os.environ then holds all those values
load_dotenv()


class Config:
    """
    Central place for all application configuration.
    All values are read from environment variables (set by .env file).
    If a variable is missing, we use a safe default value.
    """

    # --- Flask ---
    FLASK_ENV    = os.environ.get("FLASK_ENV", "development")
    FLASK_PORT   = int(os.environ.get("FLASK_PORT", 5000))
    SECRET_KEY   = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

    # --- MySQL Database ---
    DB_HOST      = os.environ.get("DB_HOST", "localhost")
    DB_PORT      = int(os.environ.get("DB_PORT", 3306))
    DB_NAME      = os.environ.get("DB_NAME", "itsm_hybrid_rag")
    DB_USER      = os.environ.get("DB_USER", "root")
    DB_PASSWORD  = os.environ.get("DB_PASSWORD", "")

    # --- Optional: LLM ---
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

    # --- Retrieval ---
    TOP_K        = int(os.environ.get("TOP_K", 10))
    CANDIDATE_K  = int(os.environ.get("CANDIDATE_K", 20))

    @classmethod
    def has_llm(cls):
        """Returns True if an LLM API key is configured."""
        return bool(cls.GEMINI_API_KEY)

    @classmethod
    def has_database(cls):
        """Returns True if a database password is configured."""
        return bool(cls.DB_PASSWORD)

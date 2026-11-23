"""
loader.py
---------
Loads all retrieval components ONCE and keeps them in memory.

Why do we do this?
- Loading BGE-M3 model takes ~5-10 seconds.
- Loading the FAISS index (22 MB) takes ~1 second.
- If Flask loaded them on every API request, every search would be very slow.
- Instead, we load everything ONCE when the Flask app starts,
  and then reuse the same objects for every request.
- This is a standard pattern in production ML serving.

Usage:
  from src.retrieval.loader import load_all
  components = load_all()
  # Then pass components to hybrid_search()
"""

import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_all(indexes_dir: str = None):
    """
    Load BM25 index, FAISS index, ticket DataFrame, and BGE-M3 model.

    Parameters:
    -----------
    indexes_dir : Path to the folder containing the index files.
                  If None, uses the default 'indexes/' folder
                  relative to the project root.

    Returns:
    --------
    A dictionary with keys:
        'faiss_index' : FAISS index object
        'bm25_index'  : BM25Okapi object
        'ticket_df'   : Pandas DataFrame of training tickets
        'model'       : SentenceTransformer (BGE-M3) model
    """

    # Determine the indexes directory
    if indexes_dir is None:
        # Build path relative to this file's location
        # This file is at: src/retrieval/loader.py
        # indexes/ is 2 levels up from here (project root)
        this_file = os.path.abspath(__file__)
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(this_file)
            )
        )
        indexes_dir = os.path.join(project_root, "indexes")

    print(f"[Loader] Loading indexes from: {indexes_dir}")

    # --- Load FAISS index ---
    faiss_path = os.path.join(indexes_dir, "faiss_index.bin")
    print("[Loader] Loading FAISS index...")
    faiss_index = faiss.read_index(faiss_path)
    print(f"[Loader] FAISS index loaded. Total vectors: {faiss_index.ntotal}")

    # --- Load BM25 index ---
    bm25_path = os.path.join(indexes_dir, "bm25_index.pkl")
    print("[Loader] Loading BM25 index...")
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)
    print("[Loader] BM25 index loaded.")

    # --- Load ticket DataFrame ---
    ticket_path = os.path.join(indexes_dir, "ticket_data.pkl")
    print("[Loader] Loading ticket data...")
    ticket_df = pd.read_pickle(ticket_path)
    print(f"[Loader] Ticket data loaded. Total tickets: {len(ticket_df)}")

    # --- Load BGE-M3 model ---
    # This takes the longest — ~5-10 seconds on first run
    # (downloads model if not cached, then loads from cache)
    print("[Loader] Loading BGE-M3 model (this may take a moment)...")
    model = SentenceTransformer("BAAI/bge-m3")
    print("[Loader] BGE-M3 model loaded successfully.")

    return {
        "faiss_index": faiss_index,
        "bm25_index":  bm25_index,
        "ticket_df":   ticket_df,
        "model":       model,
    }

"""
hybrid_search.py
----------------
Combines FAISS semantic search and BM25 keyword search
using Reciprocal Rank Fusion (RRF).

Why RRF instead of weighted average?
- BM25 scores and FAISS scores use completely different scales.
  BM25 might return scores like [14.5, 11.3, 9.2]
  FAISS might return scores like [0.91, 0.87, 0.85]
- Combining these directly (e.g., 0.5 * BM25 + 0.5 * FAISS) is problematic
  because BM25 scores will dominate (they're 10x larger).
- RRF avoids this by ignoring the actual score value.
  It only uses the RANK position (1st, 2nd, 3rd...).
- Formula: RRF_score = 1 / (rank + k)  where k=60 is the smoothing constant.

This module is the heart of the Hybrid RAG system.
"""

import pandas as pd
from .faiss_search import faiss_search
from .bm25_search import bm25_search


def reciprocal_rank_fusion(
    faiss_results: list,
    bm25_results: list,
    k: int = 60
) -> dict:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    Parameters:
    -----------
    faiss_results : List of (index, score) from FAISS — sorted best first.
    bm25_results  : List of (index, score) from BM25 — sorted best first.
    k             : Smoothing constant. Default 60 is the standard value
                    from the original RRF paper (Cormack et al., 2009).

    Returns:
    --------
    A dict mapping {doc_index: rrf_score} for all documents found in either list.
    """

    rrf_scores = {}

    # Process FAISS ranked list
    # rank starts at 1 (1st place, 2nd place, etc.)
    for rank, (doc_index, _score) in enumerate(faiss_results, start=1):
        rrf_contribution = 1.0 / (rank + k)

        if doc_index not in rrf_scores:
            rrf_scores[doc_index] = 0.0

        rrf_scores[doc_index] += rrf_contribution

    # Process BM25 ranked list (add to existing scores if doc already seen)
    for rank, (doc_index, _score) in enumerate(bm25_results, start=1):
        rrf_contribution = 1.0 / (rank + k)

        if doc_index not in rrf_scores:
            rrf_scores[doc_index] = 0.0

        rrf_scores[doc_index] += rrf_contribution

    return rrf_scores


def format_ticket(ticket_row: pd.Series, score: float) -> dict:
    """
    Convert a DataFrame row (one ticket) into a clean Python dictionary.
    This is the format that Flask will send to the frontend as JSON.

    Parameters:
    -----------
    ticket_row : A single row from the ticket DataFrame (pandas Series).
    score      : The RRF score for this ticket.

    Returns:
    --------
    A dictionary with the ticket's key fields.
    """

    # Helper to safely handle NaN (missing) values
    def safe_str(value):
        if pd.isna(value):
            return None
        return str(value).strip()

    return {
        "id": safe_str(ticket_row.get("ID")),
        "title": safe_str(ticket_row.get("Title")),
        "description": safe_str(ticket_row.get("Description")),
        "resolution": safe_str(ticket_row.get("Resolution Comments")),
        "classification": safe_str(ticket_row.get("Classification")),
        "group": safe_str(ticket_row.get("Group")),
        "service_category": safe_str(ticket_row.get("Service Category")),
        "priority": safe_str(ticket_row.get("Priority")),
        "status": safe_str(ticket_row.get("Call Status")),
        "score": round(score, 6),   # RRF score (for display/debugging)
    }


def hybrid_search(
    query: str,
    faiss_index,
    bm25_index,
    model,
    ticket_df: pd.DataFrame,
    top_k: int = 10,
    candidate_k: int = 20
) -> list:
    """
    Main function: Run hybrid retrieval and return top-k formatted tickets.

    Strategy:
    1. Get top candidate_k results from FAISS (semantic search).
    2. Get top candidate_k results from BM25 (keyword search).
    3. Fuse the two ranked lists using RRF.
    4. Sort by RRF score and return top_k results.
    5. Format each result as a clean dictionary.

    Parameters:
    -----------
    query          : The user's IT problem description.
    faiss_index    : Loaded FAISS index object.
    bm25_index     : Loaded BM25Okapi object.
    model          : Loaded SentenceTransformer (BGE-M3) model.
    ticket_df      : The training ticket DataFrame (from ticket_data.pkl).
    top_k          : Number of final results to return.
    candidate_k    : How many candidates to fetch from each retriever
                     before fusing. More candidates = better recall.

    Returns:
    --------
    List of ticket dicts, sorted by RRF score (highest first).
    Each dict has: id, title, description, resolution, classification,
                   group, service_category, priority, status, score.
    """

    # Step 1: Run both searches with candidate_k candidates each
    faiss_results = faiss_search(query, faiss_index, model, top_k=candidate_k)
    bm25_results  = bm25_search(query, bm25_index, top_k=candidate_k)

    # Step 2: Fuse using RRF
    rrf_scores = reciprocal_rank_fusion(faiss_results, bm25_results)

    # Step 3: Sort all candidates by RRF score (descending)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 4: Take only top_k results
    top_results = ranked[:top_k]

    # Step 5: Fetch actual ticket data and format as dicts
    formatted = []

    for doc_index, rrf_score in top_results:

        # Safety check: make sure index is valid
        if doc_index < 0 or doc_index >= len(ticket_df):
            continue

        ticket_row = ticket_df.iloc[doc_index]
        formatted.append(format_ticket(ticket_row, rrf_score))

    return formatted

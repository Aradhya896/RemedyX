"""
faiss_search.py
---------------
Performs semantic (vector) search using the FAISS index.

How it works:
1. Convert the user query into an embedding (a list of numbers) using BGE-M3.
2. Normalize the embedding so we can use cosine similarity.
3. Ask FAISS: "which stored vectors are closest to this query vector?"
4. Return the matching tickets with their similarity scores.

BGE-M3 turns text into a 1024-dimensional vector.
Similar texts produce similar vectors — that's the core idea of semantic search.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def faiss_search(query: str, faiss_index, model: SentenceTransformer, top_k: int = 20):
    """
    Search the FAISS index for tickets semantically similar to the query.

    Parameters:
    -----------
    query       : The user's IT problem as a string.
    faiss_index : The loaded FAISS index object.
    model       : The loaded SentenceTransformer model (BGE-M3).
    top_k       : How many results to return.

    Returns:
    --------
    A list of tuples: [(faiss_index_position, similarity_score), ...]
    The list is sorted by score, highest first.

    Example return:
    [(42, 0.91), (107, 0.87), (3, 0.85), ...]
    """

    # Step 1: Turn the query text into a numeric vector
    # model.encode() returns a numpy array of shape (1, 1024) for BGE-M3
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Step 2: Normalize the vector to unit length
    # This converts dot product similarity into cosine similarity
    # FAISS IndexFlatIP (Inner Product) works as cosine similarity
    # only when vectors are normalized — which we did during indexing too
    faiss.normalize_L2(query_embedding)

    # Step 3: Search the FAISS index
    # faiss_scores: similarity scores (higher = more similar)
    # faiss_indices: positions in the index (match rows in ticket_data.pkl)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, top_k)

    # Step 4: Package results as (index_position, score) pairs
    results = []

    for i in range(len(faiss_indices[0])):
        idx = faiss_indices[0][i]
        score = float(faiss_scores[0][i])

        # FAISS returns -1 for empty slots — skip them
        if idx == -1:
            continue

        results.append((int(idx), score))

    # Already sorted by FAISS — highest score first
    return results

"""
bm25_search.py
--------------
Performs keyword-based search using BM25 (Best Match 25).

How BM25 works (simple explanation):
- BM25 looks at which words in the query appear in each document.
- It rewards documents that contain rare query terms (TF-IDF idea).
- It penalizes very long documents to avoid unfair advantage.
- Example: "user account locked" → BM25 finds tickets with these exact words.

BM25 is great for exact keyword matches.
FAISS is great for meaning/intent matches.
Together (hybrid) they cover both cases.
"""

import numpy as np
from rank_bm25 import BM25Okapi


def bm25_search(query: str, bm25: BM25Okapi, top_k: int = 20):
    """
    Search the BM25 index for tickets matching the query keywords.

    Parameters:
    -----------
    query  : The user's IT problem as a string.
    bm25   : The loaded BM25Okapi index object.
    top_k  : How many results to return.

    Returns:
    --------
    A list of tuples: [(faiss_index_position, bm25_score), ...]
    The list is sorted by score, highest first.

    Note: The position here is the row index in the ticket DataFrame.
    It matches the same positions as the FAISS index (both built from train_data.csv).

    Example return:
    [(330, 14.5), (22, 11.3), (88, 9.7), ...]
    """

    # Step 1: Tokenize the query (simple whitespace split, lowercase)
    # BM25 works at the word level, so we split the sentence into words
    # Example: "User account locked" → ["user", "account", "locked"]
    tokenized_query = query.lower().split()

    # Step 2: Get BM25 scores for every document in the corpus
    # This returns an array of length = number of training tickets
    # Each value is the BM25 relevance score for that document
    bm25_scores = bm25.get_scores(tokenized_query)

    # Step 3: Find the indices of the top_k highest scores
    # np.argsort sorts ascending, [::-1] reverses to descending,
    # then [:top_k] takes the first top_k items
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # Step 4: Package results as (index_position, score) pairs
    results = []

    for idx in top_indices:
        score = float(bm25_scores[idx])

        # Only include results with a positive score
        # (zero score means no keyword overlap at all)
        if score > 0:
            results.append((int(idx), score))

    # Already sorted by score, highest first
    return results

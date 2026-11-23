"""
test_retrieval.py
-----------------
Quick manual test to verify the hybrid retrieval pipeline works end-to-end.

Run this from the RemedyX project root:
    cd C:/Users/aradh/OneDrive/Desktop/RemedyX
    python test_retrieval.py

Expected output:
- Loads all indexes and model
- Runs 2 test queries
- Prints top 5 results for each
- Each result shows ticket ID, description, resolution, and RRF score
"""

import sys
import os

# Add project root to Python path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.loader import load_all
from src.retrieval.hybrid_search import hybrid_search


def print_results(query, results):
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for i, ticket in enumerate(results, start=1):
        print(f"\n--- Result #{i} ---")
        print(f"Ticket ID   : {ticket['id']}")
        print(f"RRF Score   : {ticket['score']}")
        print(f"Priority    : {ticket['priority']}")
        print(f"Status      : {ticket['status']}")
        print(f"Category    : {ticket['service_category']}")

        # Show first 200 characters of description to keep output readable
        desc = ticket['description'] or "N/A"
        print(f"Description : {desc[:200]}...")

        resolution = ticket['resolution'] or "No resolution available"
        print(f"Resolution  : {resolution[:200]}...")


def main():
    # -------------------------------------------
    # Step 1: Load all components
    # -------------------------------------------
    print("Loading retrieval components...")
    components = load_all()

    faiss_index = components["faiss_index"]
    bm25_index  = components["bm25_index"]
    ticket_df   = components["ticket_df"]
    model       = components["model"]

    print("\n[OK] All components loaded successfully!")
    print(f"   Total tickets in index: {len(ticket_df)}")

    # -------------------------------------------
    # Step 2: Run test queries
    # -------------------------------------------

    test_queries = [
        "user account is locked and cannot login",
        "network connectivity issue, unable to access internet",
    ]

    for query in test_queries:
        results = hybrid_search(
            query=query,
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            model=model,
            ticket_df=ticket_df,
            top_k=5,
            candidate_k=20
        )
        print_results(query, results)

    print("\n\n[OK] Retrieval test completed successfully!")


if __name__ == "__main__":
    main()

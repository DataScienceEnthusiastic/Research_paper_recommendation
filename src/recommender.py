# recommender.py
# ========================================
# Given a paper, find the most similar papers
# by comparing their vector representations.
# ========================================
# Core logic:
#   For each pair of papers, compute cosine similarity
#   between their TF-IDF (or Sentence-BERT) vectors.
#
# Cosine similarity = (A . B) / (||A|| * ||B||)
#   = dot product of A and B divided by product of their lengths
#   = 1.0 when vectors point in the exact same direction
#   = 0.0 when vectors are perpendicular (unrelated)
#   = -1.0 when vectors point opposite directions
#
# For academic text, we expect most scores between 0.0 and 1.0.
# ========================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TOP_K_DEFAULT


class Recommender:
    """
    Takes a feature matrix (papers x features) and a DataFrame
    with paper metadata, then finds similar papers on demand.
    """

    def __init__(self, feature_matrix, papers_df):
        """
        feature_matrix: numpy array or scipy sparse matrix
                        shape = (n_papers, n_features)
        papers_df: pandas DataFrame with at least 'id' and 'title' columns
        """
        self.matrix = feature_matrix
        self.papers_df = papers_df
        self.n_papers = len(papers_df)

        # Build a lookup: paper_id -> index position in the matrix
        self.id_to_index = pd.Series(
            index=papers_df["id"].values,
            data=papers_df.index.values,
        ).to_dict()

        print(f"[recommender] Ready. {self.n_papers:,} papers in index.")

    def recommend(self, paper_id, top_k=TOP_K_DEFAULT):
        """
        Find the top_k most similar papers to the given paper_id.
        Returns a list of dicts: [{id, title, abstract (truncated), authors, categories, score}, ...]
        """
        if paper_id not in self.id_to_index:
            raise ValueError(f"Paper ID '{paper_id}' not found in the dataset.")

        # Get the index of the query paper
        query_idx = self.id_to_index[paper_id]

        # Compute cosine similarity between query and ALL papers
        # cosine_similarity returns a 2D array: [[score1, score2, ...]]
        query_vector = self.matrix[query_idx:query_idx + 1]
        all_scores = cosine_similarity(query_vector, self.matrix)[0]

        # Sort by score descending
        # We start from index 1 (not 0) because index 0 is the paper itself
        # (always has score 1.0 since it's identical to itself)
        sorted_indices = np.argsort(all_scores)[::-1][1:top_k + 1]

        results = []
        for idx in sorted_indices:
            paper = self.papers_df.iloc[idx]
            abstract = str(paper.get("abstract", ""))
            results.append({
                "id": paper["id"],
                "title": paper["title"],
                "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                "authors": paper.get("authors", ""),
                "categories": paper.get("categories", ""),
                "score": round(float(all_scores[idx]), 4),
            })

        return results

    def get_paper_by_id(self, paper_id):
        """Get full details for a single paper by its ID."""
        if paper_id not in self.id_to_index:
            return None

        idx = self.id_to_index[paper_id]
        paper = self.papers_df.iloc[idx]
        return {
            "id": paper["id"],
            "title": paper["title"],
            "abstract": paper.get("abstract", ""),
            "authors": paper.get("authors", ""),
            "categories": paper.get("categories", ""),
        }

    def search_by_keyword(self, query, top_k=20):
        """
        Simple keyword search over paper titles and abstracts.
        Uses pandas string matching (case-insensitive).
        This is NOT the recommendation engine — it's just a helper
        to let users find a starting paper by typing a topic name.
        """
        query_lower = query.lower().strip()

        # Search in both title and abstract
        title_match = self.papers_df["title"].str.lower().str.contains(
            query_lower, na=False
        )
        abstract_match = self.papers_df["abstract"].str.lower().str.contains(
            query_lower, na=False
        )

        mask = title_match | abstract_match
        results = self.papers_df[mask].head(top_k)

        matched = []
        for _, paper in results.iterrows():
            abstract = str(paper.get("abstract", ""))
            matched.append({
                "id": paper["id"],
                "title": paper["title"],
                "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                "authors": paper.get("authors", ""),
                "categories": paper.get("categories", ""),
            })

        return matched
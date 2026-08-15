# evaluator.py
# ========================================
# Measures how good our recommendations are.
# ========================================
# We use 4 metrics:
#
# 1) Precision@K
#    Of the top-K recommendations, what fraction share the same
#    ArXiv category as the query paper?
#    This tells us: "are we recommending papers from the same field?"
#
# 2) Mean Reciprocal Rank (MRR)
#    For each query, find the rank of the FIRST paper that shares
#    the query's category. Take 1/rank. Average over all queries.
#    This tells us: "how quickly does a user find something relevant?"
#
# 3) Diversity
#    Average pairwise cosine DISTANCE among recommendations.
#    (distance = 1 - similarity)
#    Interpretation: Moderate diversity (0.4-0.7) is usually ideal.
#    Very low diversity (< 0.3) means all recommendations are near-duplicates.
#    Very high diversity (> 0.8) means recommendations are barely related
#    to each other — possibly random.
#    This metric tells us about recommendation BREADTH, not quality.
#
# 4) Catalog Coverage
#    What fraction of the entire corpus appears in at least one
#    recommendation list? Low coverage means we're just recommending
#    the same popular papers to everyone.
# ========================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from src.config import TOP_K_DEFAULT, EVAL_RESULTS_PATH, PROCESSED_DIR


class Evaluator:
    """
    Runs recommendation quality benchmarks.

    LIMITATION: We use the paper's FIRST ArXiv category as the relevance signal.
    ArXiv papers often have multiple categories (e.g., cs.LG stat.ML cs.AI).
    Two papers may have different first categories but be highly related
    (e.g., cs.CL and cs.LG when both are about transformers).
    This means our precision and MRR numbers are CONSERVATIVE — actual
    topical relevance is likely higher than what we measure here.
    """

    def __init__(self, recommender, papers_df, feature_matrix):
        self.recommender = recommender
        self.papers_df = papers_df
        self.matrix = feature_matrix

        # Extract the primary category for each paper
        self.papers_df["primary_category"] = (
            self.papers_df["categories"]
            .fillna("unknown")
            .str.split()
            .str[0]
        )

    def evaluate(self, query_ids=None, n_queries=100, top_k=TOP_K_DEFAULT):
        """
        Evaluate the recommender on a set of query papers.

        Args:
            query_ids: list of paper IDs to use as queries (random sample if None)
            n_queries: number of random queries to run
            top_k: number of recommendations per query

        Returns:
            dict with metric names and values
        """
        print(f"[evaluator] Evaluating {n_queries} random queries (top-{top_k})...")

        # If no specific query IDs given, sample random papers
        if query_ids is None:
            query_ids = np.random.choice(
                self.papers_df["id"].values,
                size=min(n_queries, len(self.papers_df)),
                replace=False,
            )

        precisions = []
        reciprocal_ranks = []
        all_recommended_ids = set()

        for qid in query_ids:
            try:
                recs = self.recommender.recommend(qid, top_k=top_k)
            except (ValueError, KeyError):
                continue

            if not recs:
                continue

            # Get query paper's primary category
            query_idx = self.recommender.id_to_index.get(qid)
            if query_idx is None:
                continue
            query_category = self.papers_df.iloc[query_idx]["primary_category"]

            # Track recommended paper IDs for coverage
            rec_ids = [r["id"] for r in recs]
            all_recommended_ids.update(rec_ids)

            # ---- Precision@K ----
            # Count how many recommendations share the query's category
            match_count = 0
            for rid in rec_ids:
                rec_idx = self.recommender.id_to_index.get(rid)
                if rec_idx is not None:
                    rec_cat = self.papers_df.iloc[rec_idx]["primary_category"]
                    if rec_cat == query_category:
                        match_count += 1

            precision = match_count / top_k
            precisions.append(precision)

            # ---- MRR ----
            # Find the rank of the FIRST matching recommendation
            for rank, rid in enumerate(rec_ids, start=1):
                rec_idx = self.recommender.id_to_index.get(rid)
                if rec_idx is not None:
                    rec_cat = self.papers_df.iloc[rec_idx]["primary_category"]
                    if rec_cat == query_category:
                        reciprocal_ranks.append(1.0 / rank)
                        break
            else:
                reciprocal_ranks.append(0.0)

        # ---- Diversity ----
        # Average pairwise cosine distance among ALL recommendations
        diversity = self._compute_diversity(query_ids, top_k)

        # ---- Catalog Coverage ----
        coverage = len(all_recommended_ids) / len(self.papers_df)

        results = {
            "num_queries": len(precisions),
            f"precision@{top_k}": round(np.mean(precisions), 4),
            f"precision@{top_k}_std": round(np.std(precisions), 4),
            "mrr": round(np.mean(reciprocal_ranks), 4),
            "diversity": round(diversity, 4),
            "catalog_coverage": round(coverage, 4),
            "top_k": top_k,
        }

        return results

    def _compute_diversity(self, query_ids, top_k):
        """Compute average pairwise cosine distance among recommended paper vectors."""
        all_rec_indices = set()

        for qid in query_ids:
            try:
                recs = self.recommender.recommend(qid, top_k=top_k)
            except (ValueError, KeyError):
                continue

            for r in recs:
                idx = self.recommender.id_to_index.get(r["id"])
                if idx is not None:
                    all_rec_indices.add(idx)

        if len(all_rec_indices) < 2:
            return 0.0

        rec_indices = list(all_rec_indices)
        rec_vectors = self.matrix[rec_indices]

        # cosine_distances = 1 - cosine_similarity
        distances = cosine_distances(rec_vectors)
        # Upper triangle (excluding diagonal) gives unique pairwise distances
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        return float(np.mean(upper_tri))

    def compare_methods(self, other_matrix, method_a_name="TF-IDF", method_b_name="Sentence-BERT", n_queries=100, top_k=TOP_K_DEFAULT):
        """
        Compare two feature methods side-by-side.

        Both methods recommend from the same paper list, so the
        only difference is the feature representation used for
        similarity computation.
        """
        print(f"\n{'='*60}")
        print(f"[evaluator] Comparing {method_a_name} vs {method_b_name}")
        print(f"{'='*60}")

        # Create a temporary recommender for method B
        other_recommender = _make_recommender(other_matrix, self.papers_df)
        # Temporarily replace and evaluate
        original_recommender = self.recommender

        # Evaluate method A
        self.recommender = original_recommender
        results_a = self.evaluate(n_queries=n_queries, top_k=top_k)

        # Evaluate method B
        self.recommender = other_recommender
        results_b = self.evaluate(n_queries=n_queries, top_k=top_k)

        self.recommender = original_recommender

        # Print comparison
        print(f"\n  {'Metric':<25} {method_a_name:<15} {method_b_name:<15}")
        print(f"  {'-'*55}")
        for metric in ["precision", "mrr", "diversity", "catalog_coverage"]:
            key = f"precision@{top_k}" if metric == "precision" else metric
            val_a = results_a.get(key, 0)
            val_b = results_b.get(key, 0)
            print(f"  {metric:<25} {val_a:<15} {val_b:<15}")

        print(f"\n[Note: Higher precision/MRR/coverage is better.]")
        print(f"[Note: Diversity shows recommendation breadth: very low = too narrow, very high = too scattered.]")

        return results_a, results_b

    def save_results(self, results, filename=None):
        """Save evaluation results to CSV."""
        if filename is None:
            filename = EVAL_RESULTS_PATH

        import os
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df = pd.DataFrame([results])
        df.to_csv(filename, index=False)
        print(f"[evaluator] Saved results to: {filename}")


# Import inside the method to avoid circular imports
def _make_recommender(matrix, papers_df):
    """Factory function to avoid circular import at module level."""
    from src.recommender import Recommender
    return Recommender(matrix, papers_df)
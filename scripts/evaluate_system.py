# evaluate_system.py
# ========================================
# Standalone evaluation script.
# ========================================
# Run:  python scripts/evaluate_system.py
#
# Compares TF-IDF vs Sentence-BERT on:
#   - Precision@K
#   - Mean Reciprocal Rank (MRR)
#   - Diversity of recommendations
#   - Catalog Coverage
#
# Uses the paper's primary ArXiv category as the
# relevance signal for precision/MRR.
# ========================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import PROCESSED_DIR
from src.preprocessor import load_cleaned_data
from src.featurizer import TfidfFeaturizer, SentenceBERTFeaturizer
from src.recommender import Recommender
from src.evaluator import Evaluator

# For printing tables
def print_comparison(results_a, results_b, name_a="TF-IDF", name_b="Sentence-BERT"):
    """Print side-by-side comparison of evaluation results."""
    print(f"\n{'Metric':<25} {name_a:<15} {name_b:<15} {'Better':<10}")
    print(f"{'-'*65}")

    metrics = [
        ("precision", False),
        ("mrr", False),
        ("diversity", False),
        ("catalog_coverage", False),
    ]

    for key, higher_is_better in metrics:
        if key == "precision":
            actual_key = f"precision@{results_a.get('top_k', 10)}"
        else:
            actual_key = key

        val_a = results_a.get(actual_key, 0)
        val_b = results_b.get(actual_key, 0)

        if higher_is_better:
            winner = name_a if val_a > val_b else name_b if val_b > val_a else "Tie"
        else:
            winner = name_b if val_a < val_b else name_a if val_b < val_a else "Tie"

        print(f"  {actual_key:<23} {val_a:<15.4f} {val_b:<15.4f} {winner}")

    print(f"\n  Notes:")
    print(f"    Precision/MRR:      higher is better (more relevant recommendations)")
    print(f"    Diversity:          moderate is ideal (~0.4-0.7 is good breadth without being random)")
    print(f"    Coverage:           higher is better (broader range of papers recommended)")


def main():
    print("=" * 60)
    print("  System Evaluation")
    print("=" * 60)

    # Load data
    print("\n[1] Loading paper data...")
    df = load_cleaned_data()
    if df is None:
        print("ERROR: No cleaned data found. Run 'python run.py prepare' first.")
        sys.exit(1)

    # Load feature matrices
    print("[2] Loading TF-IDF features...")
    tfidf = TfidfFeaturizer()
    if not TfidfFeaturizer.exists():
        print("ERROR: No TF-IDF matrix found. Run 'python run.py prepare' first.")
        sys.exit(1)
    tfidf.load()

    print("[3] Loading Sentence-BERT features...")
    sbert = SentenceBERTFeaturizer()
    if not SentenceBERTFeaturizer.exists():
        print("  (Sentence-BERT not available, skipping comparison)")
        sbert_available = False
    else:
        sbert.load()
        sbert_available = True

    # Run evaluation in multiple configurations
    top_k_values = [5, 10, 20]
    n_queries = 200

    print(f"\n{'='*60}")
    print(f"  Evaluating on {n_queries} random queries")
    print(f"  Papers in corpus: {len(df):,}")
    print(f"{'='*60}")

    for top_k in top_k_values:
        print(f"\n{'='*60}")
        print(f"  Top-{top_k} Evaluation")
        print(f"{'='*60}")

        # Evaluate TF-IDF
        rec_tfidf = Recommender(tfidf.matrix, df)
        evaluator = Evaluator(rec_tfidf, df, tfidf.matrix)
        results_tfidf = evaluator.evaluate(n_queries=n_queries, top_k=top_k)

        print(f"\n  TF-IDF Results (top-{top_k}):")
        print(f"    Precision@{top_k}:    {results_tfidf[f'precision@{top_k}']:.4f}")
        print(f"    MRR:                {results_tfidf['mrr']:.4f}")
        print(f"    Diversity:          {results_tfidf['diversity']:.4f}")
        print(f"    Catalog Coverage:   {results_tfidf['catalog_coverage']:.4f}")

        # Evaluate Sentence-BERT if available
        if sbert_available:
            results_a, results_b = evaluator.compare_methods(
                sbert.matrix,
                method_a_name="TF-IDF",
                method_b_name="Sentence-BERT",
                n_queries=n_queries,
                top_k=top_k,
            )

    print(f"\n{'='*60}")
    print("  Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
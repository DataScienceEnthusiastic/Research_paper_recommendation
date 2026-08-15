# prepare_data.py
# ========================================
# ONE script to run the entire pipeline end-to-end.
# ========================================
# Run:  python prepare_data.py
#
# What it does:
# 1) Load raw ArXiv JSON
# 2) Clean/preprocess the text
# 3) Compute TF-IDF features (save matrix + vectorizer)
# 4) Compute Sentence-BERT features (save matrix)
# 5) Run evaluation comparing both methods
# 6) Build global clustering
# 7) Save everything so the Flask app can just load it
# ========================================

import sys
import os

# Add project root to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from src.config import SAMPLE_SIZE, PROCESSED_DIR
from src.data_loader import load_raw_arxiv_data, save_initial_csv
from src.preprocessor import preprocess_dataframe, save_cleaned_data, load_cleaned_data
from src.featurizer import TfidfFeaturizer, SentenceBERTFeaturizer
from src.recommender import Recommender
from src.evaluator import Evaluator
from src.visualizer import Visualizer


def main():
    print("=" * 60)
    print("  Research Paper Recommender — Data Preparation Pipeline")
    print("=" * 60)
    print(f"  Sample size: {SAMPLE_SIZE or 'ALL papers'}")
    print()

    # ---------------------------------------------------------------
    # STEP 1: Load data
    # ---------------------------------------------------------------
    print("\n[STEP 1/6] Loading ArXiv data...")
    t0 = time.time()

    # Try loading cached CSV first (much faster)
    cached = load_cleaned_data()
    if cached is not None:
        df = cached
        print(f"  Loaded {len(df):,} cleaned papers from cache.")
    else:
        df = load_raw_arxiv_data()
        save_initial_csv(df)
        print(f"  Loaded {len(df):,} raw papers from JSON.")

        # ---------------------------------------------------------------
        # STEP 2: Preprocess text
        # ---------------------------------------------------------------
        print("\n[STEP 2/6] Cleaning text (LaTeX removal, stopwords, lemmatization)...")
        df = preprocess_dataframe(df)
        save_cleaned_data(df)

    print(f"  Time: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # STEP 3: TF-IDF features
    # ---------------------------------------------------------------
    print("\n[STEP 3/6] Computing TF-IDF features...")
    t0 = time.time()

    tfidf = TfidfFeaturizer()
    if TfidfFeaturizer.exists():
        print("  TF-IDF files already exist. Loading...")
        tfidf.load()
    else:
        tfidf.fit(df["final_text"].tolist())
        tfidf.save()

    print(f"  TF-IDF matrix shape: {tfidf.matrix.shape}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # STEP 4: Sentence-BERT features
    # ---------------------------------------------------------------
    print("\n[STEP 4/6] Computing Sentence-BERT embeddings...")
    t0 = time.time()

    sbert = SentenceBERTFeaturizer()
    if SentenceBERTFeaturizer.exists():
        print("  Sentence-BERT files already exist. Loading...")
        sbert.load()
    else:
        sbert.fit(df["final_text"].tolist())
        sbert.save()

    print(f"  Sentence-BERT matrix shape: {sbert.matrix.shape}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # STEP 5: Evaluation
    # ---------------------------------------------------------------
    print("\n[STEP 5/6] Evaluating recommendation quality...")
    t0 = time.time()

    # Build recommenders for both methods
    rec_tfidf = Recommender(tfidf.matrix, df)
    evaluator = Evaluator(rec_tfidf, df, tfidf.matrix)

    print("\n--- Evaluating TF-IDF ---")
    results_tfidf = evaluator.evaluate(n_queries=100, top_k=10)
    print(f"  Precision@10: {results_tfidf['precision@10']:.4f}")
    print(f"  MRR:          {results_tfidf['mrr']:.4f}")
    print(f"  Diversity:    {results_tfidf['diversity']:.4f}")
    print(f"  Coverage:     {results_tfidf['catalog_coverage']:.4f}")

    print("\n--- Evaluating Sentence-BERT ---")
    results_a, results_b = evaluator.compare_methods(
        sbert.matrix,
        method_a_name="TF-IDF",
        method_b_name="Sentence-BERT",
        n_queries=100,
        top_k=10,
    )

    print(f"\n  Time: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # STEP 6: Global clustering
    # ---------------------------------------------------------------
    print("\n[STEP 6/6] Building global research landscape...")
    t0 = time.time()

    visualizer = Visualizer(tfidf.matrix, df)
    # Try loading existing clusters first
    if not visualizer.load():
        visualizer.fit()
        visualizer.save()

    print(f"  Time: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Data:     {PROCESSED_DIR}")
    print(f"  Papers:   {len(df):,}")
    print(f"  Clusters: {visualizer.n_clusters}")
    print("=" * 60)


if __name__ == "__main__":
    main()
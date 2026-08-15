# config.py
# ========================================
# One file for ALL settings in the project.
# Change paths, toggles, and hyperparameters here.
# No magic numbers scattered across the codebase.
# ========================================

import os
import pathlib

# --- Project Root ---
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, "data")

# --- Data Paths ---
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Raw ArXiv JSON (user must download this)
RAW_JSON_PATH = os.path.join(RAW_DIR, "arxiv-metadata-oai-snapshot.json")

# Processed CSVs
INITIAL_CSV_PATH = os.path.join(PROCESSED_DIR, "initial_df.csv")
CLEANED_CSV_PATH = os.path.join(PROCESSED_DIR, "cleaned_papers.csv")

# Saved feature matrices
TFIDF_MATRIX_PATH = os.path.join(PROCESSED_DIR, "tfidf_matrix.npz")
TFIDF_VECTORIZER_PATH = os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl")
SENTENCEBERT_MATRIX_PATH = os.path.join(PROCESSED_DIR, "sentencebert_matrix.npy")

# Saved clustering results
CLUSTER_LABELS_PATH = os.path.join(PROCESSED_DIR, "cluster_labels.npy")
CLUSTER_KEYWORDS_PATH = os.path.join(PROCESSED_DIR, "cluster_keywords.json")
TSNE_2D_PATH = os.path.join(PROCESSED_DIR, "tsne_2d.npy")

# Evaluation results
EVAL_RESULTS_PATH = os.path.join(PROCESSED_DIR, "evaluation_results.csv")

# --- Data Processing Settings ---
SAMPLE_SIZE = 10000       # How many papers to use (set to None for all)
RANDOM_SEED = 42          # Reproducibility

# --- Text Preprocessing Settings ---
MIN_WORD_LENGTH = 2       # Remove words shorter than this
MAX_ABSTRACT_WORDS = 500  # Truncate abstracts to save memory

# --- TF-IDF Settings ---
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams + bigrams (captures phrases like "machine learning")

# --- Clustering Settings ---
N_CLUSTERS_MIN = 3        # Minimum clusters to try for elbow method
N_CLUSTERS_MAX = 20       # Maximum clusters to try
SVD_N_COMPONENTS = 100    # TruncatedSVD components (must be int, unlike PCA's float variance ratio)

# --- Recommendation Settings ---
TOP_K_DEFAULT = 10       # Default number of recommendations
SIMILARITY_METRIC = "cosine"  # Distance metric for similarity

# --- Flask App Settings ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# --- Feature toggle ---
# Which feature method to use at app startup
# Options: "tfidf" or "sentencebert"
DEFAULT_FEATURE_METHOD = "tfidf"
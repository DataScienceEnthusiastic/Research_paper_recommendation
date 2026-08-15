# featurizer.py
# ========================================
# Converts cleaned text into numerical vectors
# that the recommender can compute similarity on.
# ========================================
# We support two feature methods:
#
# 1) TF-IDF (Term Frequency - Inverse Document Frequency)
#    - Each word gets a score: how often it appears in THIS paper
#      divided by how many papers it appears in overall.
#    - Words like "transformer" that are important in one paper
#      but rare overall get HIGH scores.
#    - Words like "method" that appear everywhere get LOW scores.
#    - Output: sparse matrix (one row per paper, ~10,000 columns)
#
# 2) Sentence-BERT
#    - A neural network that maps a whole sentence/paragraph
#      to a single 384-dimension vector.
#    - "Neural network" and "deep learning" will have SIMILAR vectors
#      even though they share no words (TF-IDF can't do this).
#    - Output: dense matrix (one row per paper, 384 columns)
# ========================================

import pickle
import numpy as np
import os
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MATRIX_PATH,
    TFIDF_VECTORIZER_PATH,
    SENTENCEBERT_MATRIX_PATH,
    PROCESSED_DIR,
)


# ============================================================
# TF-IDF Featurizer
# ============================================================
class TfidfFeaturizer:
    """
    TF-IDF stands for:
      TF = (number of times word appears in this paper) /
           (total words in this paper)
      IDF = log(total papers / number of papers containing this word)

    The product TF * IDF is high when a word is:
      - Frequent in THIS paper (it's important here)
      - Rare across ALL papers (it's distinctive)

    We use scikit-learn's TfidfVectorizer which does all this
    in one line. The output is a matrix where:
      - Each row = one paper
      - Each column = one word from the vocabulary
      - Each cell = that word's TF-IDF score for that paper
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",       # Remove common English words
            ngram_range=TFIDF_NGRAM_RANGE,  # 1-grams and 2-grams
            lowercase=True,
            # Only keep words made of letters, at least 2 chars
            token_pattern=r"[a-zA-Z]{2,}",
        )
        self.matrix = None

    def fit(self, texts):
        """
        Learn the vocabulary from the corpus and transform all texts.
        "Fit" means: compute IDF scores for every word in the corpus.
        "Transform" means: for each paper, compute TF * IDF for each word.
        """
        print(f"[featurizer] Fitting TF-IDF on {len(texts):,} documents...")
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"[featurizer] TF-IDF matrix shape: {self.matrix.shape}")
        print(f"[featurizer] Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self.matrix

    def save(self):
        """Save the matrix and vectorizer to disk."""
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        save_npz(TFIDF_MATRIX_PATH, self.matrix)
        print(f"[featurizer] Saved TF-IDF matrix to: {TFIDF_MATRIX_PATH}")

        with open(TFIDF_VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"[featurizer] Saved vectorizer to: {TFIDF_VECTORIZER_PATH}")

    def load(self):
        """Load previously saved matrix and vectorizer."""
        self.matrix = load_npz(TFIDF_MATRIX_PATH)
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"[featurizer] Loaded TF-IDF matrix: {self.matrix.shape}")
        return self.matrix

    @staticmethod
    def exists():
        """Check if precomputed TF-IDF files exist."""
        return os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(TFIDF_VECTORIZER_PATH)


# ============================================================
# Sentence-BERT Featurizer
# ============================================================
class SentenceBERTFeaturizer:
    """
    Sentence-BERT takes a whole sentence/paragraph and turns it
    into a single 384-dimensional vector.

    The key advantage over TF-IDF:
      - TF-IDF can only match EXACT word overlaps
      - Sentence-BERT understands SEMANTIC similarity
        "neural network" and "deep learning" -> similar vectors

    We use the 'all-MiniLM-L6-v2' model because it's:
      - Small (80MB) -> loads fast, runs on CPU
      - Fast (processes ~1000 papers/second on CPU)
      - Surprisingly good for its size
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model = None
        self.matrix = None

    def _load_model(self):
        """Lazy-load the model (import on demand to save memory)."""
        if self.model is None:
            print(f"[featurizer] Loading Sentence-BERT model: {self.MODEL_NAME}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.MODEL_NAME)
            print(f"[featurizer] Model loaded. Output dimension: {self.model.get_sentence_embedding_dimension()}")

    def fit(self, texts, batch_size=64):
        """
        Compute sentence embeddings for all texts.
        'fit' here is a misnomer (there's no training), but we
        keep the same interface as TfidfFeaturizer.
        """
        self._load_model()
        print(f"[featurizer] Computing Sentence-BERT embeddings for {len(texts):,} documents...")
        self.matrix = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        print(f"[featurizer] Sentence-BERT matrix shape: {self.matrix.shape}")
        return self.matrix

    def save(self):
        """Save the embedding matrix to disk."""
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        np.save(SENTENCEBERT_MATRIX_PATH, self.matrix)
        print(f"[featurizer] Saved Sentence-BERT matrix to: {SENTENCEBERT_MATRIX_PATH}")

    def load(self):
        """Load previously saved embedding matrix."""
        self.matrix = np.load(SENTENCEBERT_MATRIX_PATH)
        print(f"[featurizer] Loaded Sentence-BERT matrix: {self.matrix.shape}")
        return self.matrix

    @staticmethod
    def exists():
        """Check if precomputed Sentence-BERT file exists."""
        return os.path.exists(SENTENCEBERT_MATRIX_PATH)
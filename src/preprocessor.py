# preprocessor.py
# ========================================
# Cleans raw ArXiv text so our feature
# extractors get clean input.
# ========================================
# The old project skipped this step entirely.
# Garbage in = garbage out. Here we:
# 1) Remove LaTeX math ($...$, $$...$$)
# 2) Remove citation markers [1], [2,3], etc.
# 3) Remove URLs
# 4) Lowercase
# 5) Remove stopwords
# 6) Lemmatize (run -> run, running -> run, runs -> run)
# ========================================

import re
import pandas as pd
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.config import CLEANED_CSV_PATH


# Download NLTK data if not already present
def _ensure_nltk_resources():
    """Download required NLTK resources on import."""
    resources_needed = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for resource_id in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{resource_id}")
        except LookupError:
            nltk.download(resource_id, quiet=True)


_ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def remove_latex(text):
    """Remove LaTeX math expressions: $...$ and $$...$$"""
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", "", text, flags=re.DOTALL)
    return text


def remove_citations(text):
    """Remove citation markers like [1], [1,2], [1-3]"""
    text = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", text)
    return text


def remove_urls(text):
    """Remove http/https URLs"""
    text = re.sub(r"https?://\S+", "", text)
    return text


def remove_special_chars(text):
    """Keep only letters, digits, spaces, and basic punctuation."""
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\!\?\-]", " ", text)
    return text


def collapse_whitespace(text):
    """Replace multiple spaces/newlines with a single space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_text(text):
    """
    Normalize words to their base dictionary form.

    Note: NLTK's lemmatizer treats everything as a noun by default
    (no POS tagging). This means 'running' stays as 'running' instead
    of becoming 'run'. For a recommender system this is acceptable —
    the main benefit is handling plurals (papers -> paper) and
    morphological variants common in academic text.
    """
    words = text.split()
    lemmatized = [LEMMATIZER.lemmatize(word) for word in words]
    return " ".join(lemmatized)


def remove_stopwords(text):
    """Remove common English stopwords that carry little meaning."""
    words = text.split()
    filtered = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered)


def clean_text(text):
    """
    Master cleaning function: applies all steps in order.

    Each step is a separate function so you can read them individually
    and understand exactly what transformation is applied.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = remove_latex(text)
    text = remove_citations(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = collapse_whitespace(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = collapse_whitespace(text)

    return text


def preprocess_dataframe(df):
    """
    Takes a DataFrame with 'title' and 'abstract' columns.
    Returns the same DataFrame with:
      - cleaned_title
      - cleaned_abstract
      - final_text  (cleaned_title + " " + cleaned_abstract)
    """
    print(f"[preprocessor] Cleaning {len(df):,} papers...")

    # Fill NaN values to prevent errors
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")

    # Apply cleaning
    df["cleaned_title"] = df["title"].apply(clean_text)
    df["cleaned_abstract"] = df["abstract"].apply(clean_text)

    # Combine title + abstract for a single text representation
    df["final_text"] = df["cleaned_title"] + " " + df["cleaned_abstract"]

    # Collapse any whitespace created by concatenation
    df["final_text"] = df["final_text"].apply(collapse_whitespace)

    print(f"[preprocessor] Done. Sample cleaned text:")
    print(f"  Before: {df['title'].iloc[0][:80]}...")
    print(f"  After:  {df['cleaned_title'].iloc[0][:80]}...")

    return df


def save_cleaned_data(df, output_path=CLEANED_CSV_PATH):
    """Save the preprocessed DataFrame."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocessor] Saved cleaned data to: {output_path}")


def load_cleaned_data(csv_path=None):
    """Load preprocessed data if it already exists."""
    if csv_path is None:
        csv_path = CLEANED_CSV_PATH

    import os
    if not os.path.exists(csv_path):
        return None

    print(f"[preprocessor] Loading cleaned data from: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"id": str})
    print(f"[preprocessor] Loaded {len(df):,} papers")
    return df
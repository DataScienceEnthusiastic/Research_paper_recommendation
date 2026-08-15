# data_loader.py
# ========================================
# Reads the ArXiv JSON metadata file and
# turns it into a pandas DataFrame.
# ========================================
# The ArXiv dataset is a single massive JSON file
# where each LINE is one JSON record.
# We read it line-by-line so we don't crash memory.
# ========================================

import json
import pandas as pd
import os

from src.config import RAW_JSON_PATH, SAMPLE_SIZE, RANDOM_SEED


def load_raw_arxiv_data(json_path=RAW_JSON_PATH):
    """
    Read the ArXiv JSON file line by line.

    ArXiv format: each line is a JSON object with keys:
      - id: unique paper identifier (e.g., "0704.0001")
      - title: paper title
      - abstract: paper abstract
      - categories: space-separated category codes (e.g., "cs.AI cs.LG")
      - authors: author names

    We only extract the fields we actually need.
    """
    print(f"[data_loader] Reading: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"ArXiv JSON file not found at:\n{json_path}\n\n"
            "Please download it from Kaggle:\n"
            "https://www.kaggle.com/datasets/Cornell-University/arxiv\n"
            "and place it in the data/raw/ folder."
        )

    records = []
    line_count = 0

    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            # Every line is a complete JSON record
            record = json.loads(line.strip())

            records.append({
                "id": record["id"],
                "title": record["title"].strip().replace("\n", " "),
                "abstract": record["abstract"].strip().replace("\n", " "),
                "categories": record.get("categories", ""),
                "authors": record.get("authors", ""),
            })

            line_count += 1

            # Optional: stop early for fast prototyping
            if SAMPLE_SIZE and line_count >= SAMPLE_SIZE:
                break

    df = pd.DataFrame(records)
    print(f"[data_loader] Loaded {len(df):,} papers")
    return df


def save_initial_csv(df, output_path=None):
    """Save the raw (uncleaned) DataFrame for inspection."""
    if output_path is None:
        from src.config import INITIAL_CSV_PATH
        output_path = INITIAL_CSV_PATH

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[data_loader] Saved initial CSV to: {output_path}")


def load_initial_csv(csv_path=None):
    """Load a previously saved initial CSV (skip JSON parsing)."""
    if csv_path is None:
        from src.config import INITIAL_CSV_PATH
        csv_path = INITIAL_CSV_PATH

    print(f"[data_loader] Loading from CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"id": str})
    print(f"[data_loader] Loaded {len(df):,} papers")
    return df
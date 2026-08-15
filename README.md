# 📚 Research Paper Recommendation System

A modern, scalable, and interactive **Research Paper Recommendation & Discovery System** powered by NLP, Dual Feature Vectorization (TF-IDF + Sentence-BERT), Global Clustering, and an interactive Flask/Plotly Web Application.

---

## 🌟 Key Features

- **Dual Vectorization Models**:
  - **Lexical Matching (TF-IDF)**: High-performance keyword-focused vectorization.
  - **Semantic Context (Sentence-BERT)**: Dense 384-dimensional embeddings using `all-MiniLM-L6-v2` to capture semantic relationships even when papers share no overlapping words.
- **Robust Scientific Text Preprocessing**:
  - Custom regex cleaning to strip LaTeX equations (e.g., `$\alpha$`, `\cite{...}`), citations, numeric references, garbage formatting, and stop words.
  - Lemmatization via NLTK to standardize word roots.
- **Global Research Landscape & Clustering**:
  - Precomputes global paper clusters using K-Means and dimensionality reduction (PCA & t-SNE).
  - Automatically extracts cluster keywords using TF-IDF term importance.
  - Highlights query papers within the macro-level global research landscape.
- **Systematic Benchmark Evaluation**:
  - Quantitative metrics including **Precision@K**, **Mean Reciprocal Rank (MRR)**, **Recommendation Diversity**, and **Catalog Coverage**.
- **Interactive Web Interface**:
  - Built with Flask and styled with responsive CSS.
  - Full-text paper search across titles and abstracts.
  - Recommendations with similarity scores.
  - Interactive 2D Plotly cluster visualization highlighting paper positioning.

---

## 🏗️ Architecture & Pipeline Overview

```
new_paper_recommend_system/
├── data/
│   ├── raw/                        # Raw ArXiv JSON (user downloads arxiv-metadata-oai-snapshot.json)
│   └── processed/                  # Cleaned CSVs, TF-IDF matrices, SBERT embeddings, & cluster arrays
├── src/
│   ├── config.py                   # Central configuration & hyperparameters
│   ├── data_loader.py              # Memory-efficient JSON streaming & parsing
│   ├── preprocessor.py             # LaTeX removal, stopword filtering, & lemmatization
│   ├── featurizer.py               # TF-IDF & Sentence-BERT embedding generators
│   ├── recommender.py              # Cosine similarity vector search engine
│   ├── evaluator.py                # Precision@K, MRR, Diversity, & Catalog Coverage metrics
│   └── visualizer.py               # PCA, K-Means clustering, t-SNE reduction, & Plotly charts
├── app/
│   ├── routes.py                   # Flask endpoints and request routing
│   └── templates/                  # HTML templates (search, recommendations, interactive landscape)
│       ├── base.html
│       ├── home.html
│       ├── search_results.html
│       ├── paper_detail.html
│       ├── recommendations.html
│       └── cluster_view.html
├── scripts/
│   ├── prepare_data.py             # Data preparation pipeline (JSON -> Clean -> Features -> Clusters)
│   └── evaluate_system.py          # Benchmark evaluation comparing TF-IDF vs Sentence-BERT
├── requirements.txt                # Dependency specifications
├── run.py                          # Unified CLI entry point
└── README.md                       # Documentation
```

---

## 🔄 End-to-End Workflow

1. **Data Ingestion (`src/data_loader.py`)**:
   - Streams ArXiv's JSON dataset line-by-line to extract `id`, `title`, `authors`, `categories`, and `abstract` without exceeding system memory limits.
2. **Text Cleaning (`src/preprocessor.py`)**:
   - Strips math notation, normalizes text case, removes English stop words, and applies NLTK word lemmatization.
3. **Feature Extraction (`src/featurizer.py`)**:
   - Computes sparse TF-IDF feature matrices and dense Sentence-BERT document vectors.
4. **Vector Search Recommender (`src/recommender.py`)**:
   - Computes pairwise cosine similarity between query document vectors and corpus embeddings to rank top-$K$ recommendations.
5. **Clustering & Visualization (`src/visualizer.py`)**:
   - Applies PCA to preserve variance, clusters vectors using K-Means, computes 2D coordinates with t-SNE, and exports interactive Plotly figures.

---

## ⚡ Quickstart Guide

### Prerequisites

- **Python 3.10+**
- **Git**

---

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/DataScienceEnthusiastic/Research_paper_recommendation.git
cd Research_paper_recommendation

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 2. Dataset Setup

Download the ArXiv paper metadata dataset from Kaggle:
- **Dataset Link**: [ArXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Download `arxiv-metadata-oai-snapshot.json` and place it in the `data/raw/` folder:

```
data/raw/arxiv-metadata-oai-snapshot.json
```

*(Note: Pre-processed sample data matrices are already included in `data/processed/` for instant application server startup).*

---

### 3. Running the Data Pipeline (Optional for New Data)

To process raw JSON data, compute features, generate embeddings, and build clusters:

```bash
python run.py prepare
```

---

### 4. Running the Web Application

Launch the Flask web server:

```bash
python run.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🧪 System Evaluation & Benchmarking

To evaluate and compare **TF-IDF** vs **Sentence-BERT** models across standard recommendation metrics:

```bash
python scripts/evaluate_system.py
```

### Metrics Explained:

| Metric | Description | Goal |
|---|---|---|
| **Precision@K** | Ratio of top-$K$ recommended papers sharing the primary category of the query paper. | Higher is better |
| **Mean Reciprocal Rank (MRR)** | Reciprocal rank ($1 / \text{rank}$) of the first relevant recommendation. | Higher is better |
| **Diversity** | Average distance between recommended paper vectors. | Balanced (0.4 – 0.7) |
| **Catalog Coverage** | Percentage of corpus papers recommended across sample queries. | Higher is better |

---

## 👤 Author

**Uday Vijay Diware**
- GitHub: [@DataScienceEnthusiastic](https://github.com/DataScienceEnthusiastic)
- Repository: [Research_paper_recommendation](https://github.com/DataScienceEnthusiastic/Research_paper_recommendation)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

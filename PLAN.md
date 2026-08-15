# Research Paper Recommendation System - Project Plan

## The Core Idea (How We Think About It)

When a researcher reads a paper, they ask: "What else is like this?"
Our system answers that question in 3 steps:

1. **Understand the paper** - Clean the text, extract what matters
2. **Find similar papers** - Compare mathematical representations of meaning
3. **Show the landscape** - Visualize clusters of related research

Every design decision below follows from this thinking.

---

## Architecture: How The Code Mirrors Our Thinking

```
new_paper_recommend_system/
|
|-- data/                           # All data lives here
|   |-- raw/                        # Original ArXiv JSON (user downloads)
|   |-- processed/                  # Cleaned CSVs and saved matrices
|
|-- src/                            # The brain of the system
|   |-- __init__.py
|   |-- config.py                   # ONE place for all settings
|   |-- data_loader.py              # Step 1: Read raw data
|   |-- preprocessor.py             # Step 2: Clean text properly
|   |-- featurizer.py               # Step 3: Convert text to numbers
|   |-- recommender.py              # Step 4: Find similar papers
|   |-- evaluator.py                # Step 5: Measure how good we are
|   |-- visualizer.py               # Step 6: Show the clusters
|
|-- app/                            # The face of the system
|   |-- __init__.py
|   |-- routes.py                   # Flask URL handling
|   |-- templates/                  # HTML pages
|       |-- base.html
|       |-- home.html
|       |-- search_results.html
|       |-- paper_detail.html
|       |-- recommendations.html
|       |-- cluster_view.html
|
|-- scripts/                        # Run-once commands
|   |-- prepare_data.py             # Download + preprocess + featurize
|   |-- evaluate_system.py          # Run evaluation benchmarks
|
|-- requirements.txt
|-- run.py                          # Single entry point: python run.py
```

---

## Workflow: Step by Step

### Step 1: Data Loading (data_loader.py)

**What we think:** "ArXiv gives us a massive JSON file. We need to read it
without crashing our memory. We only care about: paper id, title, abstract,
categories, and authors. Nothing else."

**How the code reflects this:**
- Read JSON line-by-line (ArXiv file is ~4GB, can't load all at once)
- Extract only the 5 fields we need
- Return a clean pandas DataFrame
- User can specify how many papers to sample (default: 10,000)

### Step 2: Text Preprocessing (preprocessor.py)

**What we think (and what the old project got wrong):**
"Raw ArXiv abstracts are messy. They have LaTeX like `$\\alpha$`,
line breaks, citations like `[1]`, and garbage characters.
Before we can do ANY math on text, we must clean it.
This is the most important step. Garbage in = garbage out."

**How the code reflects this:**
1. Remove LaTeX math expressions (they carry no semantic meaning for TF-IDF/sentence embeddings)
2. Remove citation markers like [1], [2,3]
3. Remove special characters and extra whitespace
4. Lowercase everything
5. Remove stopwords (common words like "the", "is", "and" that add no meaning)
6. Lemmatize (convert "running" -> "run", "better" -> "good")
   so that different forms of the same word are treated as one concept

**Why this matters for the interviewer:**
The old project skipped all of this. We don't. Each cleaning step is a
separate function with a clear name, so anyone reading the code can see
exactly what transformations we apply and why.

### Step 3: Featurization (featurizer.py)

**What we think:**
"We need to convert human language into numbers. There are two approaches:
- TF-IDF: Counts how important a word is to a paper relative to all papers.
  Fast, interpretable, works well for keyword-heavy academic text.
- Sentence-BERT: Understands the MEANING of sentences, not just word counts.
  Slower but captures semantic similarity (e.g., 'neural network' and
  'deep learning' are related even though they share no words).

We implement BOTH so we can compare them. This shows the interviewer
we understand the tradeoff between lexical and semantic similarity."

**How the code reflects this:**
- `TfidfFeaturizer` class: fits TF-IDF, saves matrix to disk
- `SentenceBERTFeaturizer` class: computes embeddings, saves to disk
- Both produce the same interface: a matrix where each row = one paper vector
- Feature matrices are PRE-COMPUTED and loaded at app startup (not computed per request)

**Why Sentence-BERT instead of Word2Vec:**
Google News Word2Vec was trained on news articles, not academic papers.
Sentence-BERT (specifically `all-MiniLM-L6-v2`) is trained on scientific
text and produces a single 384-d vector per document (not an average of
word vectors). It's smaller, faster, and more accurate for our use case.

### Step 4: Recommendation (recommender.py)

**What we think:**
"Given a paper's vector, find the papers with the most similar vectors.
Cosine similarity measures the angle between two vectors - close to 1 means
very similar, close to 0 means unrelated, close to -1 means opposite.
For academic papers, we only care about the positive end."

**How the code reflects this:**
- `Recommender` class takes a feature matrix at init
- `recommend(paper_id, top_k)` returns the top-k most similar papers
- Similarity is computed as cosine similarity
- The query paper itself is excluded from results (it's always most similar to itself)
- Returns a list of (paper_id, similarity_score) tuples, sorted by score
- For large corpora, we use efficient dot-product computation (sklearn's cosine_similarity)

### Step 5: Evaluation (evaluator.py)

**What we think:**
"How do we know our recommendations are actually good? We can't just
 eyeball them. We need metrics:
- Precision@K: Of the top-K recommendations, what fraction share
  the same ArXiv category as the query paper? (Category is a proxy
  for topical relevance - not perfect, but reasonable.)
- Mean Reciprocal Rank (MRR): For each query, find the rank of the
  first 'relevant' recommendation, take 1/rank, average across queries.
  This tells us: how quickly does a user find something useful?
- Diversity: The average pairwise distance among recommendations.
  A good recommender doesn't just return 10 copies of the same thing.
- Catalog Coverage: What fraction of the entire corpus appears in
  SOME recommendation list? If we only ever recommend the same 100
  popular papers, coverage is low and niche papers are invisible."

**How the code reflects this:**
- `Evaluator` class takes a recommender and a DataFrame
- Evaluates on a HELD-OUT test set (not the same papers used for training)
- Computes all 4 metrics
- Can compare TF-IDF vs Sentence-BERT side by side
- Prints a clean comparison table

### Step 6: Visualization (visualizer.py)

**What we think:**
"Numbers alone don't tell the story. We want to SEE how papers cluster.
But the old project had a fundamental problem: it re-clustered papers
from scratch for EVERY query. That's not a 'landscape of research' -
that's just 'nearest neighbors rearranged.' We fix this by building
a GLOBAL clustering model once, then showing where the query paper lives
within that landscape."

**How the code reflects this:**
- Compute a global clustering of ALL papers (at startup or as a script)
- Use PCA to reduce dimensionality (preserving 95% variance)
- Determine optimal K using the elbow method + silhouette score
- Run K-Means with the chosen K
- Use t-SNE only for 2D visualization (not for clustering)
- Label each cluster with its top TF-IDF keywords (interpretable names)
- For any query paper, highlight it on the global map and show its cluster

### Step 7: Web Application (app/)

**What we think:**
"A recommender system that only works in a Jupyter notebook isn't a system.
It's a notebook. We build a proper Flask app with 4 user flows:"

**Flow 1: Search by keyword**
User types "transformer attention mechanism" -> we search titles/abstracts -> show matching papers with pagination

**Flow 2: Paper detail**
User clicks a paper -> show title, authors, abstract, categories, and a "Find Similar Papers" button

**Flow 3: Recommendations**
User clicks "Find Similar Papers" -> show top-K recommendations with similarity scores and evaluation metrics

**Flow 4: Cluster view**
User clicks "View Research Landscape" -> show interactive Plotly scatter plot of the global clustering, with the current paper highlighted

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.10+ | Standard for ML/NLP |
| Web Framework | Flask | Lightweight, sufficient |
| NLP Preprocessing | nltk + regex | Industry standard |
| Lexical Features | scikit-learn TF-IDF | Proven, interpretable |
| Semantic Features | sentence-transformers (all-MiniLM-L6-v2) | Domain-appropriate, fast |
| Similarity | scikit-learn cosine_similarity | Standard metric |
| Clustering | scikit-learn KMeans + silhouette | Standard, interpretable |
| Visualization | Plotly | Interactive, web-friendly |
| Dimensionality Reduction | scikit-learn PCA + t-SNE | Standard pipeline |
| Dataset | ArXiv (Kaggle) | Largest open academic corpus |
| Template Engine | Jinja2 (via Flask) | Standard |

---

## Output: What The User Sees

### Home Page
Clean search bar with placeholder: "Search for research papers..."
Two buttons: "Search" and "Browse Clusters"

### Search Results
List of papers showing: Title, Authors, First 150 chars of abstract, Category badge
Each paper is a clickable card leading to Paper Detail

### Paper Detail
Full title, authors, abstract, categories
Two action buttons: "Find Similar Papers" | "View on Cluster Map"

### Recommendations
Table showing: Rank, Title, Similarity Score (as %), Category match (green check or red X)
At the bottom: Evaluation metrics summary for this recommendation set

### Cluster View
Interactive Plotly scatter plot (zoom, pan, hover)
Each point = one paper, colored by cluster
Hover shows: paper title, cluster keyword label
The query paper is shown as a large red star marker
Sidebar shows cluster keyword labels and paper counts

---

## Execution Order (What We Build First)

1. `src/config.py` - All settings in one place
2. `src/data_loader.py` - Read the ArXiv data
3. `src/preprocessor.py` - Clean the text
4. `scripts/prepare_data.py` - Run steps 1-3 end to end
5. `src/featurizer.py` - TF-IDF + Sentence-BERT
6. `src/recommender.py` - Cosine similarity search
7. `src/evaluator.py` - Measure quality
8. `src/visualizer.py` - Cluster + plot
9. `app/routes.py` + templates - Web interface
10. `run.py` - Single entry point

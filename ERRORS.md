# ERRORS.md — Mistakes We Found & How We Fixed Them

This document records every conceptual mistake, bug, and bad design decision we
encountered while building this project. Each entry explains:

  1. What was wrong
  2. Why it was wrong (conceptual or practical)
  3. How we fixed it

Reading this file will help you:
- Understand why the project is structured the way it is
- Answer "what would you improve?" in an interview
- Avoid the same mistakes in your next project

---

## 1. PCA on Sparse TF-IDF Matrix (Memory Explosion)

### What we did wrong

We used `PCA(n_components=0.95)` on the TF-IDF matrix. The TF-IDF matrix is a
scipy sparse matrix (10,000 papers × 10,000 features). PCA internally calls
`.toarray()` which converts the sparse matrix to a dense NumPy array.

### Why it's wrong

A 10,000 × 10,000 dense float64 array = ~800 MB. Most laptops can't handle
this. Worse, PCA also centers the data (subtracts the mean), which creates
another dense copy. Total memory: ~2.5 GB for just this one step. The old
project accidentally worked because they were running on a small sample, but
this is wrong conceptually and practically.

### The conceptual issue

PCA is designed for dense matrices where most values are non-zero. TF-IDF
matrices are >99% zeros (sparse). The correct tool is **TruncatedSVD**, which
works directly on sparse matrices without converting to dense. TruncatedSVD
is mathematically equivalent to PCA for sparse data — it's the algorithm
behind Latent Semantic Analysis (LSA/LSI) in NLP.

### How we fixed it

```python
# Before (wrong):
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=42)
reduced = pca.fit_transform(self.matrix.toarray())  # 800MB dense array!

# After (correct):
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)  # Must be an int!
reduced = svd.fit_transform(self.matrix)  # Works on sparse matrix directly
```

Note: Unlike PCA, TruncatedSVD does NOT support a variance ratio (float)
for `n_components`. You must specify an exact integer. We use 100 components
which typically captures 85-95% of the variance for TF-IDF matrices of
this size. To verify: explained variance can be checked via
`svd.explained_variance_ratio_.sum()`.```

---

## 2. Cluster Keywords Picked Generic Words Instead of Distinctive Ones

### What we did wrong

We used TF-IDF within each cluster to find "top keywords." The code was:
```python
vec = TfidfVectorizer(max_features=500, stop_words="english")
tfidf = vec.fit_transform(cluster_texts)
scores = tfidf.sum(axis=0).A1  # Sum TF-IDF across all docs in cluster
top_indices = scores.argsort()[-10:][::-1]
```

### Why it's wrong

TF-IDF within a cluster tells you which words are frequent within that
cluster's documents. But the SAME words ("model", "method", "paper", "result")
are frequent in EVERY cluster. So every cluster would get keywords like
["model", "method", "paper", "data", "result"] — useless for distinguishing
clusters.

The problem: we were measuring **frequency within the cluster** but not
**distinctiveness across clusters**.

### The conceptual fix

We need a metric that scores words high when they appear in THIS cluster
AND low in other clusters. This is exactly what **c-TF-IDF** (class-based
TF-IDF) does:

1. Count word frequency in each cluster
2. Count word frequency across ALL clusters combined
3. Score = cluster_frequency / global_frequency

Words like "model" appear in every cluster, so their score drops.
Words like "quantum" appear mostly in one cluster, so their score is high.

```python
# For each cluster, compute a c-TF-IDF score
for word in cluster_words:
    score = cluster_freq[word] / global_freq[word]
    # Higher = more distinctive to this cluster
```

---

## 3. NLTK Resource Loading Was Fragile

### What we did wrong

We tried to download NLTK resources (stopwords, wordnet) using try/except:
```python
def _ensure_nltk_resources():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)

_ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))  # Might fail here!
```

### Why it's wrong

If NLTK data isn't downloaded, `_ensure_nltk_resources()` downloads it.
But then `stopwords.words("english")` at the module level (line 40) is a
NEW call that might fail if the download hasn't fully registered.
The try/except catches the right exception, but the module-level code
runs unconditionally after the function — there's no safety net.

### How we fixed it

Use `nltk.data.find()` which is the OFFICIAL API for checking resource
availability before loading:

```python
def _ensure_nltk_resources():
    for resource_id in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{resource_id}")
        except LookupError:
            nltk.download(resource_id, quiet=True)

_ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))  # Now safe
```

---

## 4. NLTK Lemmatizer Doesn't Handle Verbs Naturally

### What we thought

We thought `lemmatize("running")` would return `"run"`, similar to how
`lemmatize("papers")` returns `"paper"`.

### What actually happens

NLTK's `WordNetLemmatizer` defaults to treating every word as a NOUN.
For nouns, it correctly handles plurals ("papers" → "paper").
For verbs, it does nothing ("running" → "running", "wrote" → "wrote").

### Why this happens

WordNet stores different word forms under different "synsets" (meaning IDs).
The lemmatizer needs to know the Part of Speech (POS) to find the correct
lemma. Without POS tag, it assumes noun — which is the most common POS
in English but wrong for verbs.

### How we handled it

We chose NOT to add POS tagging. Here's why:

- POS tagging adds significant complexity and runtime (need to load a
  separate POS tagger model)
- For a RECOMMENDATION system, perfect lemmatization isn't critical —
  "running algorithm" and "run algorithm" are close enough in TF-IDF
  space (they share the word "algorithm")
- The noun forms that DO get lemmatized (plurals, possessives) give us
  most of the benefit

We documented this limitation explicitly in the code comment:

```python
def lemmatize_text(text):
    """
    Note: NLTK's lemmatizer treats everything as a noun by default.
    This means 'running' stays as 'running' instead of becoming 'run'.
    For a recommender system this is acceptable — the main benefit is
    handling plurals (papers -> paper) and morphological variants.
    """
```

---

## 5. Per-Query Clustering in the Old Project (Wrong Approach)

### What the old project did

Every time a user clicked "View Cluster Map", the system:
1. Took the top-1000 similar papers to the query
2. Ran PCA + K-Means + t-SNE on just those 1000 papers
3. Showed the result

### Why it's wrong

This creates a DIFFERENT clustering for every query. If you look at Paper A's
cluster map and Paper B's cluster map, they show completely different
landscapes. There's no "global research landscape" — just 1000 nearest
neighbors rearranged each time.

More importantly, it's mislabeled as "clustering" when it's really just
"nearest neighbor visualization with extra steps."

### What we do instead

We build a GLOBAL clustering ONCE:
1. Run PCA/K-Means/t-SNE on ALL papers (not per-query)
2. Every paper gets a fixed cluster label
3. For visualization, we highlight the query paper on the pre-built map
4. The landscape is the same for every query — only the highlight changes

This gives users a consistent mental model of how papers relate to each other.

---

## 6. "Diversity = Higher Is Better" Was Wrong

### What we wrote

In the evaluation output, we printed:
```
Higher diversity means less repetitive recommendations
```

### Why it's wrong

If diversity is 0.98, it means recommendations are nearly random — each one
is very different from the others. That's not good. If diversity is 0.95,
it's even worse than 0.90? No, that logic doesn't work.

Diversity is a **descriptive** metric, not a **normative** one:
- Very low (< 0.3): All recommendations are near-duplicates (bad)
- Moderate (0.4–0.7): Recommendations cover related but distinct subtopics (good)
- Very high (> 0.8): Recommendations are barely related to each other (bad)

The ideal value depends on the use case. A literature review tool might want
lower diversity (tightly related papers). A discovery tool might want higher
diversity (broad exploration).

### How we fixed it

Changed all output to:
```
Diversity: moderate is ideal (~0.4-0.7 is good breadth without being random)
```

---

## 7. Evaluation Uses First ArXiv Category Only (Conservative)

### What we do

To measure recommendation relevance, we check if the recommended paper's
first ArXiv category matches the query paper's first category.

### Why it's a limitation

ArXiv papers can have MULTIPLE categories. For example:
- "Attention Is All You Need" has categories: `cs.CL cs.LG cs.CV`

Two papers about transformers might have different first categories
(e.g., `cs.CL` and `cs.LG`) but be perfectly relevant to each other.
Our evaluation would count this as a "miss" — incorrectly.

### How we handle it

We document this limitation openly. Our precision numbers are CONSERVATIVE.
The actual topical relevance of recommendations is higher than what our
metrics show. For an interview, this shows you understand the gap between
a convenient evaluation proxy and true relevance measurement.

---

## 8. TF-IDF Gets Double Stopword Removal (Harmless But Redundant)

### What happens

1. The preprocessor removes stopwords (NLTK's list)
2. The TF-IDF vectorizer also removes stopwords (scikit-learn's list)

### Why it's not a bug

Both lists are reasonable. After step 1, the stopwords are already gone from
the text. Step 2's `stop_words="english"` has nothing left to remove.
This is harmless — just redundant.

### Why we kept it

- The preprocessor's stopword removal ALSO benefits Sentence-BERT mode
  (where the TF-IDF vectorizer isn't used)
- Having `stop_words="english"` in the TF-IDF vectorizer acts as a safety
  net in case anyone skips the preprocessor
- An interviewer might ask: "Are you aware this is redundant?" — you can
  explain the reasoning

---

## 9. Sentence-BERT Truncates Long Abstracts

### What happens

Sentence-BERT's `all-MiniLM-L6-v2` model has a maximum input length of
512 tokens (BERT's limit). ArXiv abstracts average 200-500 words, but
tokenization (subword) expands this to 300-800 tokens. Abstracts longer
than 512 tokens get TRUNCATED — the rest is lost.

### Why we accept this

For recommendation purposes, the first ~400 tokens of an academic abstract
contain:
- The paper's topic/research area
- The methodology used
- The key contribution/results

The truncated part usually contains related work citations, experimental
details, and acknowledgments — less important for similarity matching.

This is a known tradeoff: we trade perfect abstract representation for
the ability to use a fast, small embedding model.

---

## 10. Brute-Force Cosine Similarity Is O(n) Per Query

### What we do

For each recommendation request, we compute cosine similarity between the
query paper and ALL papers in the corpus (10,000 papers).

### Why this is a limitation

At 10,000 papers, this takes ~50ms — fast enough. But at 100,000 papers,
it takes ~500ms. At 1 million papers, it takes ~5 seconds — too slow
for a web application.

### How we'd fix it for production

Use Approximate Nearest Neighbor (ANN) search:
- **FAISS**: Facebook's library for billion-scale similarity search
- **Annoy**: Spotify's library for memory-bounded ANN
- **HNSWlib**: Hierarchical Navigable Small World graphs (fastest)

These trade ~5% accuracy for 100x speed by searching only a subset of
the corpus. For a recommendation system, this accuracy loss is invisible
to users.

---

---

## 11. Plotly/WebGL Cluster Visualization Was Too Slow to Render

### What we did wrong

We used Plotly to render a scatter plot of 10,000 papers in the browser. The
approach was:
1. Server sends 2MB of JSON (paper coordinates, labels, colors)
2. Browser loads Plotly.js library from CDN (~3MB download)
3. Plotly renders the scatter using WebGL
4. User can pan/zoom interactively

### Why it's wrong

Step 1 was fast (~300ms transmission), but steps 2-3 never completed in the
browser. WebGL with 10,000 labeled points on consumer hardware can take
30-60 seconds to render — or fail silently if the GPU is underpowered.

### The conceptual issue

This is a "rich client" mistake: we assumed the browser is better at rendering
than the server. For a static scatter plot (no pan/zoom needed beyond what a
static image provides), the server can generate the visualization once and
send the result as an image — no JS, no CDN, no GPU requirements.

### How we fixed it

```python
# Generate SVG on the server using matplotlib (Agg backend)
import matplotlib
matplotlib.use('Agg')  # No display needed
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
for cluster_id in sorted(unique_clusters):
    mask = labels == cluster_id
    ax.scatter(x[mask], y[mask], c=[color], s=8, label=f"Cluster {cluster_id}")
# Highlight query paper with a red star
ax.scatter([qx], [qy], c='red', marker='*', s=200, edgecolors='white')
buf = io.BytesIO()
fig.savefig(buf, format='svg', bbox_inches='tight', dpi=120)
svg = buf.getvalue().decode('utf-8')
```

The SVG is embedded directly in the HTML with `{{ cluster_svg | safe }}`.
Rendering is instant — the browser only needs to display an SVG image.

---

## Summary: What We Learned

| Lesson | Type | Severity |
|--------|------|----------|
| PCA doesn't work on sparse matrices — use TruncatedSVD | Technical | Critical |
| Cluster keywords must measure distinctiveness, not frequency | Conceptual | Medium |
| NLTK resources need proper checking before loading | Technical | Low |
| NLTK lemmatizer needs POS tags for verbs | Conceptual | Low (documented) |
| Per-query clustering is misleading — use global clustering | Conceptual | Medium |
| Diversity isn't "higher is better" — it's context-dependent | Evaluation | Medium |
| Category-based evaluation is conservative (multi-label issue) | Evaluation | Low (documented) |
| Double stopword removal is redundant but harmless | Technical | Low |
| Sentence-BERT truncates long abstracts | Technical | Low (documented) |
| Brute-force similarity doesn't scale past 100K papers | Design | Low (noted) |
| Plotly/WebGL cluster viz too slow — use server-side matplotlib SVG | Technical | Medium |
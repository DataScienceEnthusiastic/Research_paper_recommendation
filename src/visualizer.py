# visualizer.py
# ========================================
# Builds a GLOBAL cluster map of all research papers,
# then highlights where a query paper lives in that map.
# ========================================
# The old project computed clusters fresh for EVERY query.
# That was wrong — it showed a different "landscape" each time.
#
# What we do instead:
# 1) Find the best number of clusters via elbow + silhouette
# 2) Run K-Means ONCE on ALL papers (or a representative sample)
# 3) Reduce to 2D with t-SNE for visualization
# 4) Label each cluster with its top keywords
# 5) For any query paper, just highlight it on the existing map
# ========================================

import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter

from src.config import (
    N_CLUSTERS_MIN,
    N_CLUSTERS_MAX,
    SVD_N_COMPONENTS,
    CLUSTER_LABELS_PATH,
    CLUSTER_KEYWORDS_PATH,
    TSNE_2D_PATH,
    PROCESSED_DIR,
)


class Visualizer:
    """
    Creates and stores a global clustering of all papers,
    then provides methods to visualize where any paper sits.

    Important design decisions:
    - We use TruncatedSVD (not PCA) for dimensionality reduction.
      PCA requires a dense matrix which would be ~800MB for 10K papers x 10K features.
      TruncatedSVD works directly on the sparse TF-IDF matrix.
      However, TruncatedSVD (unlike PCA) requires an explicit integer n_components,
      not a variance ratio. We use {SVD_N_COMPONENTS} components which typically captures
      85-95% of variance for TF-IDF matrices of this size.
    - t-SNE is only for visualization. The cluster positions in 2D do NOT
      represent actual cluster boundaries — they're a 2D "sketch" that
      roughly preserves local neighborhoods.
    - Cluster keywords are computed using a simplified c-TF-IDF approach:
      words are scored by how frequent they are in THIS cluster relative
      to ALL clusters, so we get DISTINCTIVE keywords.
    """
    SVD_N_COMPONENTS = SVD_N_COMPONENTS  # Fixed integer — TruncatedSVD doesn't support variance ratios

    def __init__(self, feature_matrix, papers_df):
        self.matrix = feature_matrix
        self.papers_df = papers_df
        self.n_clusters = None
        self.labels = None
        self.centroids = None
        self.tsne_2d = None
        self.cluster_keywords = {}
        self.is_fitted = False

    def find_optimal_clusters(self, max_clusters=N_CLUSTERS_MAX):
        """
        Use the elbow method + silhouette score to find the best K.

        How the elbow method works:
        - For each K, compute the sum of squared distances from each point
          to its cluster center (inertia).
        - Plot inertia vs K. The "elbow" is where the curve bends
          (diminishing returns on adding more clusters).

        How silhouette works:
        - For each point, measures how similar it is to its OWN cluster
          vs the NEXT nearest cluster.
        - Ranges from -1 (wrong cluster) to +1 (perfect cluster).
        - We pick the K with the highest silhouette score.
        """
        # TruncatedSVD first to speed up clustering.
        # Unlike PCA, TruncatedSVD works directly on sparse matrices
        # without converting to dense (which would be ~800MB).
        print(f"[visualizer] Reducing dimensionality with TruncatedSVD to {self.SVD_N_COMPONENTS} components...")
        svd = TruncatedSVD(n_components=self.SVD_N_COMPONENTS, random_state=42)
        reduced = svd.fit_transform(self.matrix)
        print(f"[visualizer] TruncatedSVD reduced to {reduced.shape[1]} dimensions")

        inertias = []
        silhouettes = []

        for k in range(N_CLUSTERS_MIN, max_clusters + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(reduced)

            inertias.append(km.inertia_)
            sil = silhouette_score(reduced, labels)
            silhouettes.append(sil)
            print(f"  K={k:2d}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

        # Pick the K with the highest silhouette score
        best_idx = np.argmax(silhouettes)
        best_k = N_CLUSTERS_MIN + best_idx

        print(f"\n[visualizer] Optimal K (by silhouette): {best_k}")
        print(f"[visualizer] Silhouette score at K={best_k}: {silhouettes[best_idx]:.4f}")

        return best_k, inertias, silhouettes

    def fit(self, n_clusters=None):
        """
        Compute the global clustering:
        1. TruncatedSVD to reduce dimensions (works on sparse matrices)
        2. K-Means to find clusters
        3. t-SNE to create 2D projection for visualization
        4. Extract cluster keywords using c-TF-IDF
        """
        if n_clusters is None:
            n_clusters, _, _ = self.find_optimal_clusters()
        self.n_clusters = n_clusters

        print(f"[visualizer] TruncatedSVD reducing to {self.SVD_N_COMPONENTS} components...")
        self.svd = TruncatedSVD(n_components=self.SVD_N_COMPONENTS, random_state=42)
        reduced = self.svd.fit_transform(self.matrix)

        print(f"[visualizer] Running K-Means with K={n_clusters}...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(reduced)
        self.centroids = self.kmeans.cluster_centers_

        print(f"[visualizer] Computing t-SNE for 2D visualization...")
        # Perplexity of ~40 is better than 30 for datasets in the 5-10K range.
        # t-SNE only shows LOCAL structure — cluster sizes in the plot
        # do NOT reflect actual cluster sizes in the original space.
        self.tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
        self.tsne_2d = self.tsne.fit_transform(reduced)

        print(f"[visualizer] Extracting cluster keywords...")
        self._extract_cluster_keywords()

        self.is_fitted = True
        print(f"[visualizer] Clustering complete. {n_clusters} clusters across {len(self.papers_df):,} papers.")

    def _extract_cluster_keywords(self, top_n=10):
        """
        Find DISTINCTIVE keywords for each cluster using c-TF-IDF logic.

        The key insight: We don't just find the most FREQUENT words in a cluster.
        That would give us words like "model", "method", "paper" which appear
        everywhere. Instead, we find words that are:
          (a) Frequent IN this cluster, AND
          (b) Rare OUTSIDE this cluster

        We approximate this by:
        1. Counting word frequency within each cluster
        2. Counting word frequency across ALL clusters
        3. Scoring = cluster_freq / global_freq (higher = more distinctive)

        See: "c-TF-IDF" from Maarten Grootendorst's BERTopic library.
        """
        # Build vocabulary of all words across all clusters
        all_words = []
        cluster_texts_dict = {}

        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            cluster_texts = (
                self.papers_df.loc[mask, "title"].fillna("")
                + " "
                + self.papers_df.loc[mask, "abstract"].fillna("")
            )

            if len(cluster_texts) < 2:
                self.cluster_keywords[int(cluster_id)] = ["(small cluster)"]
                continue

            # Use CountVectorizer to get word counts for this cluster
            vec = CountVectorizer(
                max_features=1000,
                stop_words="english",
                token_pattern=r"[a-zA-Z]{2,}",
                lowercase=True,
            )
            try:
                count_matrix = vec.fit_transform(cluster_texts)
                # Sum across all documents in this cluster
                cluster_word_counts = np.array(count_matrix.sum(axis=0)).flatten()
                feature_names = vec.get_feature_names_out()
                cluster_texts_dict[cluster_id] = (feature_names, cluster_word_counts)
                all_words.extend(cluster_texts)
            except Exception:
                self.cluster_keywords[int(cluster_id)] = ["(keyword extraction failed)"]

        if not cluster_texts_dict:
            return

        # Compute global word frequencies across ALL clusters
        global_vec = CountVectorizer(
            max_features=2000,
            stop_words="english",
            token_pattern=r"[a-zA-Z]{2,}",
            lowercase=True,
        )
        global_counts = global_vec.fit_transform(all_words)
        global_word_counts = np.array(global_counts.sum(axis=0)).flatten()
        global_feature_names = global_vec.get_feature_names_out()
        global_freq = dict(zip(global_feature_names, global_word_counts))

        # For each cluster, score words by cluster_freq / global_freq
        for cluster_id, (feature_names, cluster_word_counts) in cluster_texts_dict.items():
            word_scores = {}
            for i, word in enumerate(feature_names):
                cluster_freq = cluster_word_counts[i]
                global_freq_val = global_freq.get(word, 1)
                # c-TF-IDF-like score: cluster_freq / global_freq
                # Add 1 to denominator to avoid division by zero
                score = cluster_freq / (global_freq_val + 1)
                word_scores[word] = score

            # Sort by score descending and take top_n
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            self.cluster_keywords[int(cluster_id)] = [w for w, _ in sorted_words[:top_n]]

    def get_cluster_of_paper(self, paper_id):
        """Return the cluster ID and keywords for a specific paper."""
        idx = self.papers_df.index[self.papers_df["id"] == paper_id].tolist()
        if not idx:
            return None, None

        cluster_id = self.labels[idx[0]]
        keywords = self.cluster_keywords.get(int(cluster_id), [])
        return int(cluster_id), keywords

    def generate_cluster_svg(self, highlight_paper_id=None, max_points=1500):
        """
        Generate the cluster scatter plot as an inline SVG string.

        Builds a minimal SVG directly with circle elements — no matplotlib,
        no plotly, no JS. Renders instantly in every browser.
        """
        n_total = len(self.papers_df)

        cluster_colors = [
            '#4361ee', '#f72585', '#7209b7', '#3a0ca3', '#4cc9f0',
            '#e63946', '#2d6a4f', '#d4a373', '#e76f51', '#264653',
            '#8338ec', '#ff6b6b', '#48bfe3', '#f4a261', '#2a9d8f',
            '#9c89b8', '#ef476f', '#06d6a0', '#118ab2', '#073b4c',
        ]

        # Pick a highlight point index
        highlight_idx = None
        if highlight_paper_id:
            idx = self.papers_df.index[self.papers_df["id"] == highlight_paper_id].tolist()
            if idx:
                highlight_idx = idx[0]

        # Downsample preserving cluster proportions
        if n_total > max_points:
            rng = np.random.RandomState(42)
            sampled = []
            for cid in range(self.n_clusters):
                ci = np.where(self.labels == cid)[0]
                prop = len(ci) / n_total
                want = max(1, int(max_points * prop))
                if len(ci) > want:
                    chosen = rng.choice(ci, size=want, replace=False).tolist()
                else:
                    chosen = ci.tolist()
                sampled.extend(chosen)
            if highlight_idx is not None and highlight_idx not in sampled:
                sampled.append(highlight_idx)
            sampled = sorted(sampled)
            plot_x = self.tsne_2d[sampled, 0]
            plot_y = self.tsne_2d[sampled, 1]
            plot_labels = self.labels[sampled]
            hl_plot_idx = sampled.index(highlight_idx) if highlight_idx in sampled else -1
        else:
            plot_x = self.tsne_2d[:, 0]
            plot_y = self.tsne_2d[:, 1]
            plot_labels = self.labels
            sampled = list(range(n_total))
            hl_plot_idx = sampled.index(highlight_idx) if highlight_idx is not None else -1

        # Map cluster IDs to keyword labels
        kw_map = {cid: ", ".join(kws[:4]) for cid, kws in self.cluster_keywords.items()}

        # Normalize coordinates to fit in an 800x500 SVG viewport with padding
        xs = np.array(plot_x, dtype=float)
        ys = np.array(plot_y, dtype=float)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1

        w_view, h_view = 800, 540
        margin_l, margin_r = 60, 20
        margin_t, margin_b = 40, 60

        def norm_x(x):
            return margin_l + (x - (x_min - pad_x)) / ((x_max - x_min) + 2 * pad_x) * (w_view - margin_l - margin_r)
        def norm_y(y):
            return margin_t + (y - (y_min - pad_y)) / ((y_max - y_min) + 2 * pad_y) * (h_view - margin_t - margin_b)

        # Build SVG parts
        svg_parts = []
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {w_view} {h_view}" '
            f'style="width: 100%; height: auto; display: block;">'
        )
        svg_parts.append('<rect width="100%" height="100%" fill="#ffffff" />')
        svg_parts.append(f'<text x="{w_view // 2}" y="22" '
                         f'text-anchor="middle" font-size="14" font-weight="bold" '
                         f'fill="#1a1a2e">Research Paper Landscape</text>')

        # Scatter points per cluster
        for cid in range(self.n_clusters):
            mask = plot_labels == cid
            if not mask.any():
                continue
            color = cluster_colors[cid % len(cluster_colors)]
            pts_x = xs[mask]
            pts_y = ys[mask]
            for px, py in zip(pts_x, pts_y):
                cx = norm_x(px)
                cy = norm_y(py)
                svg_parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" '
                    f'fill="{color}" fill-opacity="0.6" />'
                )

        # Highlight query paper
        if hl_plot_idx >= 0:
            hx = norm_x(xs[hl_plot_idx])
            hy = norm_y(ys[hl_plot_idx])
            svg_parts.append(
                f'<polygon points="{hx},{hy-8} {hx+3},{hy-2.5} {hx+8},{hy-2.5} '
                f'{hx+4.5},{hy+1.5} {hx+6},{hy+7} {hx},{hy+3.5} '
                f'{hx-6},{hy+7} {hx-4.5},{hy+1.5} {hx-8},{hy-2.5} '
                f'{hx-3},{hy-2.5}" fill="red" stroke="#333" stroke-width="1" />'
            )

        # Legend
        svg_parts.append(
            f'<g transform="translate(20, {h_view - 100})">'
        )
        leg_cols = 4
        leg_items = []
        for cid in range(self.n_clusters):
            if not (plot_labels == cid).any():
                continue
            color = cluster_colors[cid % len(cluster_colors)]
            label = kw_map.get(int(cid), f"Cluster {cid}")
            leg_items.append((color, label))
        for i, (color, label) in enumerate(leg_items):
            col = i % leg_cols
            row = i // leg_cols
            lx = col * 195
            ly = row * 16
            svg_parts.append(
                f'<circle cx="{lx + 4}" cy="{ly + 6}" r="3" fill="{color}" fill-opacity="0.7" />'
            )
            svg_parts.append(
                f'<text x="{lx + 10}" y="{ly + 9}" font-size="9" fill="#555" '
                f'style="font-family: sans-serif;">{label}</text>'
            )
        svg_parts.append('</g>')

        # Note about sampling
        note = f"Showing {len(sampled):,} of {n_total:,} papers"
        if n_total > max_points:
            svg_parts.append(
                f'<text x="{w_view // 2}" y="{h_view - 12}" '
                f'text-anchor="middle" font-size="10" fill="#888" '
                f'style="font-family: sans-serif;">{note}</text>'
            )

        # Selected paper star label
        if hl_plot_idx >= 0:
            svg_parts.append(
                f'<text x="{w_view - margin_r - 5}" y="{margin_t + 12}" '
                f'font-size="9" fill="#888" text-anchor="end" '
                f'style="font-family: sans-serif;">★ = Selected Paper</text>'
            )

        svg_parts.append('</svg>')
        return "\n".join(svg_parts), len(sampled), n_total

    def save(self):
        """Save clustering results to disk."""
        import os
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        np.save(CLUSTER_LABELS_PATH, self.labels)
        np.save(TSNE_2D_PATH, self.tsne_2d)

        # Save cluster keywords as JSON (convert numpy types to native)
        keywords_serializable = {
            str(k): v for k, v in self.cluster_keywords.items()
        }
        with open(CLUSTER_KEYWORDS_PATH, "w") as f:
            json.dump(keywords_serializable, f, indent=2)

        print(f"[visualizer] Saved clustering results to: {PROCESSED_DIR}")

    def load(self, n_clusters=None):
        """Load previously saved clustering results."""
        import os

        if not os.path.exists(CLUSTER_LABELS_PATH):
            print("[visualizer] No saved clusters found. Run .fit() first.")
            return False

        self.labels = np.load(CLUSTER_LABELS_PATH)
        self.tsne_2d = np.load(TSNE_2D_PATH)
        self.n_clusters = n_clusters or len(np.unique(self.labels))

        with open(CLUSTER_KEYWORDS_PATH, "r") as f:
            raw = json.load(f)
            self.cluster_keywords = {int(k): v for k, v in raw.items()}

        self.is_fitted = True
        print(f"[visualizer] Loaded {self.n_clusters} clusters for {len(self.labels):,} papers.")
        return True
# routes.py
# ========================================
# All Flask URL routes for the web application.
# ========================================

from flask import Blueprint, render_template, request, redirect, url_for, jsonify, make_response
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TOP_K_DEFAULT, DEFAULT_FEATURE_METHOD
from src.data_loader import load_initial_csv
from src.preprocessor import load_cleaned_data
from src.featurizer import TfidfFeaturizer, SentenceBERTFeaturizer
from src.recommender import Recommender
from src.visualizer import Visualizer

main_bp = Blueprint("main", __name__)

# Global state — loaded once when the app starts
_recommender = None
_visualizer = None
_papers_df = None


def init_app():
    """Load all precomputed data into memory. Called once at startup."""
    global _recommender, _visualizer, _papers_df

    # Load paper data
    _papers_df = load_cleaned_data()
    if _papers_df is None:
        # Fallback: load raw initial CSV and create final_text
        _papers_df = load_initial_csv()
        _papers_df["title"] = _papers_df["title"].fillna("")
        _papers_df["abstract"] = _papers_df["abstract"].fillna("")
        _papers_df["final_text"] = _papers_df["title"] + " " + _papers_df["abstract"]

    print(f"[app] Loaded {len(_papers_df):,} papers")

    # Load feature matrix (whichever method is configured as default)
    if DEFAULT_FEATURE_METHOD == "tfidf":
        featurizer = TfidfFeaturizer()
        featurizer.load()
        print(f"[app] Using TF-IDF features: {featurizer.matrix.shape}")
    else:
        featurizer = SentenceBERTFeaturizer()
        featurizer.load()
        print(f"[app] Using Sentence-BERT features: {featurizer.matrix.shape}")

    _recommender = Recommender(featurizer.matrix, _papers_df)

    # Load or build visualizer
    _visualizer = Visualizer(featurizer.matrix, _papers_df)
    if not _visualizer.load():
        print("[app] No precomputed clusters found. Clustering now...")
        _visualizer.fit()
        _visualizer.save()

    print("[app] Ready!")


# ============================================================
# ROUTES
# ============================================================

@main_bp.route("/")
def home():
    """Home page with search bar."""
    # Show some random papers as examples
    sample_papers = _papers_df.sample(10)[["id", "title", "authors", "categories"]].to_dict("records")
    return render_template("home.html", papers=sample_papers)


@main_bp.route("/search")
def search():
    """Search papers by keyword."""
    query = request.args.get("q", "").strip()
    if not query:
        return redirect(url_for("main.home"))

    results = _recommender.search_by_keyword(query, top_k=50)
    return render_template("search_results.html", query=query, results=results)


@main_bp.route("/paper/<paper_id>")
def paper_detail(paper_id):
    """Show full details of a paper."""
    paper = _recommender.get_paper_by_id(paper_id)
    if paper is None:
        return render_template("error.html", message=f"Paper '{paper_id}' not found.")

    return render_template("paper_detail.html", paper=paper)


@main_bp.route("/recommend/<paper_id>")
def recommend(paper_id):
    """Get recommendations for a paper."""
    try:
        top_k = int(request.args.get("k", TOP_K_DEFAULT))
    except ValueError:
        top_k = TOP_K_DEFAULT

    try:
        paper = _recommender.get_paper_by_id(paper_id)
        if paper is None:
            return render_template("error.html", message=f"Paper '{paper_id}' not found.")

        recommendations = _recommender.recommend(paper_id, top_k=top_k)

        return render_template(
            "recommendations.html",
            paper=paper,
            recommendations=recommendations,
            method=DEFAULT_FEATURE_METHOD.upper(),
        )
    except Exception as e:
        return render_template("error.html", message=str(e))


@main_bp.route("/cluster/<paper_id>")
def cluster_view(paper_id):
    """Show the global cluster map with this paper highlighted.

    The cluster plot is generated as an inline SVG on the server.
    No JavaScript, no AJAX, no CDN — renders instantly in every browser.
    """
    paper = _recommender.get_paper_by_id(paper_id)
    if paper is None:
        return render_template("error.html", message=f"Paper '{paper_id}' not found.")

    cluster_id, cluster_keywords = _visualizer.get_cluster_of_paper(paper_id)
    svg, points_shown, total_papers = _visualizer.generate_cluster_svg(
        highlight_paper_id=paper_id, max_points=1500
    )

    response = make_response(render_template(
        "cluster_view.html",
        paper=paper,
        cluster_id=cluster_id,
        cluster_keywords=", ".join(cluster_keywords[:8]) if cluster_keywords else "Unknown",
        cluster_svg=svg,
        points_shown=points_shown,
        total_papers=total_papers,
    ))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@main_bp.route("/api/recommend/<paper_id>")
def api_recommend(paper_id):
    """JSON API endpoint for recommendations."""
    try:
        top_k = int(request.args.get("k", TOP_K_DEFAULT))
    except ValueError:
        top_k = TOP_K_DEFAULT

    try:
        recs = _recommender.recommend(paper_id, top_k=top_k)
        return jsonify({"status": "ok", "recommendations": recs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 404
# run.py
# ========================================
# Single entry point for the entire project.
# ========================================
# How to use:
#
#   1) FIRST TIME - Prepare the data:
#      python run.py prepare
#
#   2) EVERY TIME - Start the web app:
#      python run.py
#
# What happens in each mode:
#   "prepare": runs the full data pipeline (JSON -> cleaned text -> features -> clusters)
#   "server":  starts the Flask web app (default)
# ========================================

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        # Run data preparation pipeline
        print("=" * 60)
        print("  Running data preparation pipeline...")
        print("=" * 60)
        from scripts.prepare_data import main as prepare_main
        prepare_main()
        return

    # Start the Flask web app
    print("=" * 60)
    print("  Starting Research Paper Recommender")
    print("  Open: http://localhost:5000")
    print("=" * 60)

    from app.routes import main_bp, init_app
    from flask import Flask
    from src.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

    # Point Flask to the app/templates folder
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    app = Flask(__name__, template_folder=os.path.join(app_dir, "templates"))
    app.register_blueprint(main_bp)

    # Load precomputed data at startup
    print("[run] Initializing app (loading data, features, clusters)...")
    init_app()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)


if __name__ == "__main__":
    main()
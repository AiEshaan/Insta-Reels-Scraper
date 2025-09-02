import os
import pandas as pd
from flask import Flask, request, send_file, jsonify
# Removed render_template import since we're API-only
from dotenv import load_dotenv
from main import run_agent
from scheduler import scheduler
from functools import wraps

load_dotenv()
app = Flask(__name__)

OUTPUT_DIR = "output"
CSV_FILE = os.path.join(OUTPUT_DIR, "saved_reels.csv")
XLSX_FILE = os.path.join(OUTPUT_DIR, "saved_reels.xlsx")

API_KEY = os.getenv('API_KEY', 'your-secret-api-key')

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != API_KEY:
            return jsonify({'error': 'API key is required'}), 401
        return f(*args, **kwargs)
    return decorated_function


def read_latest():
    """Read whichever file (CSV/XLSX) is freshest"""
    if os.path.exists(XLSX_FILE) and (
        not os.path.exists(CSV_FILE)
        or os.path.getmtime(XLSX_FILE) > os.path.getmtime(CSV_FILE)
    ):
        return pd.read_excel(XLSX_FILE)
    elif os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["Reel URL", "Caption", "Thumbnail"])


@app.route("/api/start", methods=["POST"])
@require_api_key
def start():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400
            
        os.environ["IG_USERNAME"] = username
        os.environ["IG_PASSWORD"] = password
        
        # Run initial scraping
        run_agent(username=username, password=password)
        
        # Start automatic scraping scheduler
        scheduler.start(username=username, password=password)
        
        return jsonify({
            "message": "Successfully started automatic scraping",
            "status": "running",
            "interval_hours": scheduler.interval_hours
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
@require_api_key
def status():
    reels = read_latest().to_dict("records")
    return jsonify({
        "running": scheduler.is_running,
        "interval_hours": scheduler.interval_hours,
        "reels_count": len(reels)
    })


@app.route("/api/refresh", methods=["POST"])
@require_api_key
def refresh():
    try:
        run_agent()
        return jsonify({"message": "Successfully refreshed reels"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<fmt>")
@require_api_key
def download(fmt):
    try:
        if fmt == "csv" and os.path.exists(CSV_FILE):
            return send_file(CSV_FILE, as_attachment=True, download_name="saved_reels.csv")
        elif fmt == "excel" and os.path.exists(XLSX_FILE):
            return send_file(XLSX_FILE, as_attachment=True, download_name="saved_reels.xlsx")
        return jsonify({"error": "No file available for download"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
@require_api_key
def stop():
    scheduler.stop()
    return jsonify({"message": "Successfully stopped automatic scraping"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
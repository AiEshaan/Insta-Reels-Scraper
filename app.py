import os
import pandas as pd
import asyncio
import json
from flask import Flask, request, send_file, jsonify, render_template_string
from dotenv import load_dotenv
from main import run_agent
from scheduler import scheduler
from functools import wraps
from ai_agent_main import IntelligentReelsAgent

load_dotenv()
app = Flask(__name__)

OUTPUT_DIR = "output"
CSV_FILE = os.path.join(OUTPUT_DIR, "scrapped_reels.csv")
XLSX_FILE = os.path.join(OUTPUT_DIR, "scrapped_reels.xlsx")

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
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        if fmt == "csv" and os.path.exists(CSV_FILE):
            download_path = os.path.join(downloads_path, "scrapped_reels.csv")
            df = pd.read_csv(CSV_FILE)
            df.to_csv(download_path, index=False)
            return jsonify({"message": f"File saved to {download_path}"})
        elif fmt == "excel" and os.path.exists(XLSX_FILE):
            download_path = os.path.join(downloads_path, "scrapped_reels.xlsx")
            df = pd.read_excel(XLSX_FILE)
            df.to_excel(download_path, index=False, engine="openpyxl")
            return jsonify({"message": f"File saved to {download_path}"})
        return jsonify({"error": "No file available for download"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
@require_api_key
def stop():
    try:
        scheduler.shutdown()
        return jsonify({"message": "Scraping stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai-agent/start", methods=["POST"])
@require_api_key
def start_ai_agent():
    """Start the intelligent AI agent"""
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        
        # Run AI agent in background
        def run_ai_agent_background():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            agent = IntelligentReelsAgent()
            results = loop.run_until_complete(agent.run_intelligent_agent(username, password))
            
            # Save results for later retrieval
            with open('output/latest_ai_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            loop.close()
        
        import threading
        thread = threading.Thread(target=run_ai_agent_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "AI agent started successfully",
            "status": "running",
            "note": "Check /api/ai-agent/status for progress"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai-agent/status")
@require_api_key
def ai_agent_status():
    """Get AI agent status and results"""
    try:
        # Check if results file exists
        results_file = 'output/latest_ai_results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            return jsonify({
                "status": "completed" if results.get('success') else "failed",
                "results": results,
                "timestamp": results.get('timestamp'),
                "summary": results.get('summary', {}),
                "insights": results.get('insights', []),
                "recommendations": results.get('recommendations', [])
            })
        else:
            return jsonify({
                "status": "running",
                "message": "AI agent is still processing..."
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai-agent/results")
@require_api_key
def ai_agent_results():
    """Get detailed AI agent results"""
    try:
        results_file = 'output/ai_agent_results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({
                "status": "no_results",
                "message": "No AI agent results available yet. Please start the AI agent first.",
                "suggestion": "Use the 'Start AI Agent' button or POST to /api/ai-agent/start to begin analysis."
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai-agent/memory")
@require_api_key
def ai_agent_memory():
    """Get AI agent memory summary"""
    try:
        memory_file = 'output/memory_summary.json'
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                memory = json.load(f)
            return jsonify(memory)
        else:
            return jsonify({
                "status": "no_memory",
                "message": "No AI agent memory data available yet. Please start the AI agent first.",
                "suggestion": "The AI agent builds memory as it processes content and learns from interactions."
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Simple HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Instagram Reels AI Agent</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 10px 0; }
        .ai-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        .ai-button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .ai-button:hover { background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%); }
        input { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .ai-status { background: #e3f2fd; color: #0d47a1; }
        .tabs { display: flex; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #ddd; cursor: pointer; border-radius: 4px 4px 0 0; margin-right: 5px; }
        .tab.active { background: #007bff; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <h1>🤖 Instagram Reels AI Agent</h1>
    <p>Intelligent content scraping, analysis, and autonomous actions powered by AI</p>
    
    <div class="tabs">
        <div class="tab active" onclick="showTab('basic')">Basic Scraper</div>
        <div class="tab" onclick="showTab('ai')">AI Agent</div>
    </div>

    <div id="basic" class="tab-content active">
        <div class="container">
            <h3>📋 Basic Scraper</h3>
            <p>Traditional Instagram reels scraping functionality</p>
            <button onclick="checkStatus()">Check Status</button>
            <div id="statusResult"></div>
        </div>
    </div>

    <div id="ai" class="tab-content">
        <div class="container ai-container">
            <h3>🤖 AI Agent Features</h3>
            <p>Intelligent content analysis, learning, and autonomous actions</p>
            <ul>
                <li>🧠 <strong>Content Analysis:</strong> Sentiment, emotion, and viral potential assessment</li>
                <li>📊 <strong>Trend Detection:</strong> Identify trending topics and hashtags</li>
                <li>🎯 <strong>Autonomous Actions:</strong> Smart filtering, categorization, and recommendations</li>
                <li>💾 <strong>Learning Memory:</strong> Adapts to user preferences over time</li>
                <li>✨ <strong>Content Generation:</strong> AI-powered captions and hashtags</li>
            </ul>
            <button class="ai-button" onclick="startAIAgent()">🚀 Start AI Agent</button>
            <button class="ai-button" onclick="checkAIStatus()">📊 Check AI Status</button>
            <button class="ai-button" onclick="viewAIResults()">📈 View Results</button>
            <div id="aiStatusResult"></div>
        </div>
    </div>


    
    <div class="container">
        <h3>🚀 Getting Started</h3>
        <ol>
            <li><strong>Basic Scraper:</strong> Use the traditional scraping functionality</li>
            <li><strong>AI Agent:</strong> Experience intelligent content analysis and autonomous actions</li>
            <li>Check the <code>output/</code> folder for all generated data and reports</li>
        </ol>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }

        async function checkStatus() {
            try {
                const response = await fetch('/api/status', {
                    headers: {
                        'X-API-Key': 'reel-scraper-2024-secret'
                    }
                });
                const data = await response.json();
                document.getElementById('statusResult').innerHTML = 
                    '<div class="status success">Status: ' + JSON.stringify(data, null, 2) + '</div>';
            } catch (error) {
                document.getElementById('statusResult').innerHTML = 
                    '<div class="status error">Error: ' + error.message + '</div>';
            }
        }

        async function startAIAgent() {
            try {
                // For demo purposes, using placeholder credentials
                // In production, you'd want a proper form for this
                const response = await fetch('/api/ai-agent/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': 'reel-scraper-2024-secret'
                    },
                    body: JSON.stringify({
                        username: 'demo_user',
                        password: 'demo_pass'
                    })
                });
                const data = await response.json();
                document.getElementById('aiStatusResult').innerHTML = 
                    '<div class="status ai-status">🚀 ' + data.message + '<br>Status: ' + data.status + '</div>';
            } catch (error) {
                document.getElementById('aiStatusResult').innerHTML = 
                    '<div class="status error">Error: ' + error.message + '</div>';
            }
        }

        async function checkAIStatus() {
            try {
                const response = await fetch('/api/ai-agent/status', {
                    headers: {
                        'X-API-Key': 'reel-scraper-2024-secret'
                    }
                });
                const data = await response.json();
                
                let html = '<div class="status ai-status">';
                html += '<h4>🤖 AI Agent Status: ' + data.status + '</h4>';
                
                if (data.summary) {
                    html += '<p><strong>📊 Summary:</strong></p>';
                    html += '<ul>';
                    html += '<li>Content Scraped: ' + (data.summary.total_content_scraped || 0) + '</li>';
                    html += '<li>Content Analyzed: ' + (data.summary.total_content_analyzed || 0) + '</li>';
                    html += '<li>Patterns Learned: ' + (data.summary.patterns_learned || 0) + '</li>';
                    html += '<li>Actions Taken: ' + (data.summary.actions_taken || 0) + '</li>';
                    html += '<li>Content Generated: ' + (data.summary.content_generated || 0) + '</li>';
                    html += '</ul>';
                }
                
                if (data.insights && data.insights.length > 0) {
                    html += '<p><strong>💡 Key Insights:</strong></p>';
                    html += '<ul>';
                    data.insights.forEach(insight => {
                        html += '<li>' + insight + '</li>';
                    });
                    html += '</ul>';
                }
                
                html += '</div>';
                document.getElementById('aiStatusResult').innerHTML = html;
                
            } catch (error) {
                document.getElementById('aiStatusResult').innerHTML = 
                    '<div class="status error">Error: ' + error.message + '</div>';
            }
        }

        async function viewAIResults() {
            try {
                const response = await fetch('/api/ai-agent/results', {
                    headers: {
                        'X-API-Key': 'reel-scraper-2024-secret'
                    }
                });
                const data = await response.json();
                
                if (data.status === 'no_results') {
                    document.getElementById('aiStatusResult').innerHTML = 
                        '<div class="status ai-status">ℹ️ ' + data.message + '<br><small>' + data.suggestion + '</small></div>';
                } else {
                    // Open results in new window for better viewing
                    const newWindow = window.open('', '_blank');
                    newWindow.document.write(`
                        <html>
                            <head><title>AI Agent Results</title></head>
                            <body style="font-family: Arial, sans-serif; padding: 20px;">
                                <h1>🤖 AI Agent Results</h1>
                                <pre style="background: #f5f5f5; padding: 20px; border-radius: 8px; overflow: auto;">
${JSON.stringify(data, null, 2)}
                                </pre>
                            </body>
                        </html>
                    `);
                    
                    document.getElementById('aiStatusResult').innerHTML = 
                        '<div class="status success">📈 Results opened in new window</div>';
                }
                    
            } catch (error) {
                document.getElementById('aiStatusResult').innerHTML = 
                    '<div class="status error">Error: ' + error.message + '</div>';
            }
        }
    </script>
</body>
</html>
'''

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
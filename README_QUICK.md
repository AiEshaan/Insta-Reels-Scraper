# Instagram Saved Reels Agent - Quick Start

1. Create and activate a Python virtualenv:
   ```bash
   python -m venv venv
   # activate it
   pip install -r requirements.txt
   playwright install
   ```

2. Run the Flask app:
   ```bash
   python app.py
   ```

3. Open http://127.0.0.1:5000 in your browser.

4. Enter an Instagram username & password and click "Log In & Scrape Saved".
   - The browser will open and the agent will run.
   - If Instagram asks for verification (2FA/challenge) you'll have to complete it manually in the opened browser.

5. After scraping finishes, you'll be redirected to the dashboard where you can view results and download the Excel/CSV.

## Extra recommendations & safety notes

### 2FA
The script cannot fully automate 2FA flows. If IG challenges the login, run in headful mode and manually complete the challenge before continuing.

### Rate-limiting
Don't run too frequently. Add scheduling (e.g., schedule library) and long intervals.

Use a secondary account if you care about avoiding restrictions on your main account.

### Cookies/session reuse
The agent saves output/storage_<username>.json after login. Reuse reduces repeated logins and reduces suspicion.

### Headful for troubleshooting
Run with headless=False (default in app.py call) to visually inspect what's happening.

### Logging
main.py logs to console. You can add logging.FileHandler to persist logs.

## Troubleshooting tips (common errors)

- Can't instantiate abstract class AgentManager — don't import or instantiate abstract LangChain agent classes; we didn't use them in this final code.

- @vite/client 404 errors — those are from frontend dev tooling; our templates are plain Jinja2 and do not require Vite.

- If Flask or packages are missing, ensure virtualenv activated and pip install -r requirements.txt ran.

- If Instagram displays "We suspect automated behavior", pause and manually resolve in the opened browser and then re-run with cookies saved.
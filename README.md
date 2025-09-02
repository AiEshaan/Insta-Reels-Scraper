# Instagram Saved Reels Auto-Scraper API

A headless API service that automatically scrapes your saved reels from Instagram every 6 hours.

## Authentication
All API endpoints require an API key to be passed in the request header:
```
X-API-Key: your-api-key
```

## API Endpoints

### Start Auto-Scraping
```
POST /api/start
Content-Type: application/json

{
    "username": "your_instagram_username",
    "password": "your_instagram_password"
}
```
Response:
```json
{
    "message": "Successfully started automatic scraping",
    "status": "running",
    "interval_hours": 6
}
```

### Check Status
```
GET /api/status
```
Response:
```json
{
    "running": true,
    "interval_hours": 6,
    "reels_count": 42
}
```

### Manual Refresh
```
POST /api/refresh
```
Response:
```json
{
    "message": "Successfully refreshed reels"
}
```

### Download Data
```
GET /api/download/csv   # Download as CSV
GET /api/download/excel # Download as Excel
```
Downloads the scraped reels data in the requested format.

### Stop Auto-Scraping
```
POST /api/stop
```
Response:
```json
{
    "message": "Successfully stopped automatic scraping"
}
```

## Error Responses
All endpoints return error responses in this format:
```json
{
    "error": "Error message description"
}
```

## Security
- All endpoints are local-only (127.0.0.1)
- Credentials are used locally and not stored permanently
- Data is saved to local CSV/Excel files only
from datetime import datetime
from .db import db

class SavedContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('scraping_job.id'), nullable=False)
    content_type = db.Column(db.String(20))  # video, image, carousel
    content_url = db.Column(db.String(255), nullable=False)
    caption = db.Column(db.Text)
    author = db.Column(db.String(80))
    saved_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SavedContent {self.id} - {self.content_type}>'
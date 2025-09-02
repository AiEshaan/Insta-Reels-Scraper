from datetime import datetime
from .db import db

class ScrapingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    instagram_username = db.Column(db.String(80), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    error_message = db.Column(db.Text)
    output_file = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    saved_contents = db.relationship('SavedContent', backref='job', lazy=True)
    
    def __repr__(self):
        return f'<ScrapingJob {self.id} - {self.status}>'
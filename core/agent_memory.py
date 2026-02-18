"""
Agent Memory System
Advanced memory and learning capabilities for the AI agent
"""

import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Single memory entry"""
    entry_id: str
    entry_type: str  # 'user_action', 'content_analysis', 'decision', 'feedback'
    content_id: str
    data: Dict[str, Any]
    timestamp: datetime
    importance: float  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class UserPreference:
    """User preference pattern"""
    preference_id: str
    category: str
    preference_type: str  # 'content_type', 'timing', 'quality', 'sentiment'
    value: Any
    confidence: float
    evidence_count: int
    last_updated: datetime

@dataclass
class LearningPattern:
    """Learned behavioral pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float

class AgentMemorySystem:
    """
    Advanced Memory System for AI Agent Learning
    
    Features:
    - Persistent memory storage with SQLite
    - User preference learning and adaptation
    - Pattern recognition and behavioral learning
    - Memory consolidation and forgetting
    - Context-aware memory retrieval
    - Performance tracking and optimization
    """
    
    def __init__(self, memory_db_path: str = "output/agent_memory.db"):
        self.memory_db_path = memory_db_path
        self.memory_entries = {}
        self.user_preferences = {}
        self.learning_patterns = {}
        self.short_term_memory = deque(maxlen=100)  # Recent memories
        self.working_memory = {}  # Current session context
        
        # Learning parameters
        self.learning_rate = 0.1
        self.forgetting_factor = 0.95
        self.importance_threshold = 0.3
        self.consolidation_interval = timedelta(hours=24)
        
        # Initialize database
        self._init_memory_database()
        
        # Load existing memories
        self._load_memories()
        
        logger.info("Agent Memory System initialized")
    
    def _init_memory_database(self):
        """Initialize SQLite database for persistent memory"""
        try:
            os.makedirs(os.path.dirname(self.memory_db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Memory entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    entry_type TEXT,
                    content_id TEXT,
                    data TEXT,
                    timestamp TEXT,
                    importance REAL,
                    access_count INTEGER,
                    last_accessed TEXT
                )
            ''')
            
            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_id TEXT PRIMARY KEY,
                    category TEXT,
                    preference_type TEXT,
                    value TEXT,
                    confidence REAL,
                    evidence_count INTEGER,
                    last_updated TEXT
                )
            ''')
            
            # Learning patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    description TEXT,
                    conditions TEXT,
                    outcomes TEXT,
                    confidence REAL,
                    usage_count INTEGER,
                    success_rate REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Memory database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing memory database: {e}")
    
    def _load_memories(self):
        """Load existing memories from database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            
            # Load memory entries
            df_memories = pd.read_sql_query("SELECT * FROM memory_entries", conn)
            for _, row in df_memories.iterrows():
                entry = MemoryEntry(
                    entry_id=row['entry_id'],
                    entry_type=row['entry_type'],
                    content_id=row['content_id'],
                    data=json.loads(row['data']),
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    importance=row['importance'],
                    access_count=row['access_count'],
                    last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None
                )
                self.memory_entries[entry.entry_id] = entry
            
            # Load user preferences
            df_prefs = pd.read_sql_query("SELECT * FROM user_preferences", conn)
            for _, row in df_prefs.iterrows():
                pref = UserPreference(
                    preference_id=row['preference_id'],
                    category=row['category'],
                    preference_type=row['preference_type'],
                    value=json.loads(row['value']),
                    confidence=row['confidence'],
                    evidence_count=row['evidence_count'],
                    last_updated=datetime.fromisoformat(row['last_updated'])
                )
                self.user_preferences[pref.preference_id] = pref
            
            # Load learning patterns
            df_patterns = pd.read_sql_query("SELECT * FROM learning_patterns", conn)
            for _, row in df_patterns.iterrows():
                pattern = LearningPattern(
                    pattern_id=row['pattern_id'],
                    pattern_type=row['pattern_type'],
                    description=row['description'],
                    conditions=json.loads(row['conditions']),
                    outcomes=json.loads(row['outcomes']),
                    confidence=row['confidence'],
                    usage_count=row['usage_count'],
                    success_rate=row['success_rate']
                )
                self.learning_patterns[pattern.pattern_id] = pattern
            
            conn.close()
            
            logger.info(f"Loaded {len(self.memory_entries)} memories, {len(self.user_preferences)} preferences, {len(self.learning_patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    def store_memory(self, entry_type: str, content_id: str, data: Dict[str, Any], 
                    importance: float = 0.5) -> str:
        """Store a new memory entry"""
        try:
            entry_id = f"{entry_type}_{content_id}_{datetime.now().timestamp()}"
            
            entry = MemoryEntry(
                entry_id=entry_id,
                entry_type=entry_type,
                content_id=content_id,
                data=data,
                timestamp=datetime.now(),
                importance=importance
            )
            
            # Store in memory
            self.memory_entries[entry_id] = entry
            self.short_term_memory.append(entry)
            
            # Store in database
            self._save_memory_entry(entry)
            
            logger.debug(f"Stored memory: {entry_type} for {content_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return ""
    
    def learn_from_user_action(self, content_id: str, user_action: str, 
                              content_data: Dict[str, Any], context: Dict[str, Any] = None):
        """Learn from user actions to update preferences"""
        try:
            # Store the action as memory
            action_data = {
                'action': user_action,
                'content_data': content_data,
                'context': context or {}
            }
            
            memory_id = self.store_memory('user_action', content_id, action_data, importance=0.8)
            
            # Extract learning signals
            self._extract_preference_signals(user_action, content_data, context)
            
            # Update learning patterns
            self._update_learning_patterns(user_action, content_data, context)
            
            logger.info(f"Learned from user action: {user_action} on {content_id}")
            
        except Exception as e:
            logger.error(f"Error learning from user action: {e}")
    
    def _extract_preference_signals(self, user_action: str, content_data: Dict[str, Any], 
                                   context: Dict[str, Any] = None):
        """Extract preference signals from user actions"""
        try:
            # Content type preferences
            category = content_data.get('primary_category', 'general')
            self._update_preference('content_category', category, user_action, 1.0)
            
            # Quality preferences
            filter_score = content_data.get('filter_score', 0.5)
            if user_action in ['save', 'like', 'share']:
                self._update_preference('quality_threshold', 'min_score', filter_score, 0.8)
            
            # Timing preferences
            if context and 'timestamp' in context:
                timestamp = datetime.fromisoformat(context['timestamp'])
                hour = timestamp.hour
                if user_action in ['save', 'like']:
                    self._update_preference('timing', 'preferred_hour', hour, 0.6)
            
            # Sentiment preferences
            if 'sentiment' in content_data:
                sentiment = content_data['sentiment']
                if user_action in ['save', 'like']:
                    self._update_preference('sentiment', 'preferred_sentiment', sentiment, 0.7)
            
        except Exception as e:
            logger.error(f"Error extracting preference signals: {e}")
    
    def _update_preference(self, category: str, preference_type: str, value: Any, confidence: float):
        """Update or create a user preference"""
        try:
            pref_id = f"{category}_{preference_type}"
            
            if pref_id in self.user_preferences:
                # Update existing preference
                pref = self.user_preferences[pref_id]
                
                # Weighted average for numerical values
                if isinstance(value, (int, float)) and isinstance(pref.value, (int, float)):
                    new_value = (pref.value * pref.confidence + value * confidence) / (pref.confidence + confidence)
                    new_confidence = min(pref.confidence + confidence * self.learning_rate, 1.0)
                else:
                    # For categorical values, use most confident
                    if confidence > pref.confidence:
                        new_value = value
                        new_confidence = confidence
                    else:
                        new_value = pref.value
                        new_confidence = pref.confidence
                
                pref.value = new_value
                pref.confidence = new_confidence
                pref.evidence_count += 1
                pref.last_updated = datetime.now()
                
            else:
                # Create new preference
                pref = UserPreference(
                    preference_id=pref_id,
                    category=category,
                    preference_type=preference_type,
                    value=value,
                    confidence=confidence,
                    evidence_count=1,
                    last_updated=datetime.now()
                )
                self.user_preferences[pref_id] = pref
            
            # Save to database
            self._save_user_preference(pref)
            
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
    
    def _update_learning_patterns(self, user_action: str, content_data: Dict[str, Any], 
                                 context: Dict[str, Any] = None):
        """Update learning patterns based on user behavior"""
        try:
            # Create pattern conditions
            conditions = {
                'content_category': content_data.get('primary_category', 'general'),
                'filter_score_range': self._get_score_range(content_data.get('filter_score', 0.5)),
                'time_of_day': self._get_time_period(datetime.now().hour)
            }
            
            # Create pattern outcomes
            outcomes = {
                'user_action': user_action,
                'positive_response': user_action in ['save', 'like', 'share', 'comment']
            }
            
            # Generate pattern ID
            pattern_id = f"pattern_{hash(str(conditions))}_{user_action}"
            
            if pattern_id in self.learning_patterns:
                # Update existing pattern
                pattern = self.learning_patterns[pattern_id]
                pattern.usage_count += 1
                
                # Update success rate
                if outcomes['positive_response']:
                    pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 1) / pattern.usage_count
                else:
                    pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1)) / pattern.usage_count
                
                # Update confidence
                pattern.confidence = min(pattern.confidence + 0.1, 1.0)
                
            else:
                # Create new pattern
                pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type='user_behavior',
                    description=f"User tends to {user_action} content with {conditions}",
                    conditions=conditions,
                    outcomes=outcomes,
                    confidence=0.5,
                    usage_count=1,
                    success_rate=1.0 if outcomes['positive_response'] else 0.0
                )
                self.learning_patterns[pattern_id] = pattern
            
            # Save to database
            self._save_learning_pattern(pattern)
            
        except Exception as e:
            logger.error(f"Error updating learning patterns: {e}")
    
    def get_relevant_memories(self, query_context: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Retrieve relevant memories based on context"""
        try:
            relevant_memories = []
            
            for entry in self.memory_entries.values():
                relevance_score = self._calculate_memory_relevance(entry, query_context)
                if relevance_score > 0.3:  # Relevance threshold
                    relevant_memories.append((entry, relevance_score))
                    
                    # Update access statistics
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
            
            # Sort by relevance and return top results
            relevant_memories.sort(key=lambda x: x[1], reverse=True)
            
            return [memory[0] for memory in relevant_memories[:limit]]
            
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")
            return []
    
    def get_user_preferences_for_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get user preferences relevant to current context"""
        try:
            relevant_prefs = {}
            
            for pref in self.user_preferences.values():
                if pref.confidence > 0.5:  # Only confident preferences
                    relevant_prefs[f"{pref.category}_{pref.preference_type}"] = {
                        'value': pref.value,
                        'confidence': pref.confidence,
                        'evidence_count': pref.evidence_count
                    }
            
            return relevant_prefs
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    def predict_user_response(self, content_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict user response to content based on learned patterns"""
        try:
            predictions = {
                'save_probability': 0.5,
                'like_probability': 0.5,
                'share_probability': 0.3,
                'ignore_probability': 0.4
            }
            
            # Use learning patterns to make predictions
            content_conditions = {
                'content_category': content_data.get('primary_category', 'general'),
                'filter_score_range': self._get_score_range(content_data.get('filter_score', 0.5)),
                'time_of_day': self._get_time_period(datetime.now().hour)
            }
            
            matching_patterns = []
            for pattern in self.learning_patterns.values():
                if self._patterns_match(pattern.conditions, content_conditions):
                    matching_patterns.append(pattern)
            
            # Aggregate predictions from matching patterns
            if matching_patterns:
                for action in ['save', 'like', 'share', 'ignore']:
                    action_patterns = [p for p in matching_patterns if p.outcomes.get('user_action') == action]
                    if action_patterns:
                        weighted_success = sum(p.success_rate * p.confidence for p in action_patterns)
                        total_weight = sum(p.confidence for p in action_patterns)
                        predictions[f"{action}_probability"] = weighted_success / total_weight if total_weight > 0 else 0.5
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting user response: {e}")
            return {'save_probability': 0.5, 'like_probability': 0.5, 'share_probability': 0.3, 'ignore_probability': 0.4}
    
    def consolidate_memories(self):
        """Consolidate and optimize memory storage"""
        try:
            current_time = datetime.now()
            
            # Apply forgetting to old, unimportant memories
            memories_to_remove = []
            for entry_id, entry in self.memory_entries.items():
                age_days = (current_time - entry.timestamp).days
                
                # Calculate decay factor
                decay_factor = self.forgetting_factor ** age_days
                adjusted_importance = entry.importance * decay_factor
                
                # Remove if importance falls below threshold
                if adjusted_importance < self.importance_threshold and age_days > 30:
                    memories_to_remove.append(entry_id)
                else:
                    # Update importance
                    entry.importance = adjusted_importance
            
            # Remove forgotten memories
            for entry_id in memories_to_remove:
                del self.memory_entries[entry_id]
                self._remove_memory_from_db(entry_id)
            
            # Consolidate similar preferences
            self._consolidate_preferences()
            
            logger.info(f"Memory consolidation completed. Removed {len(memories_to_remove)} old memories")
            
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
    
    def _calculate_memory_relevance(self, memory: MemoryEntry, context: Dict[str, Any]) -> float:
        """Calculate relevance score between memory and context"""
        try:
            relevance = 0.0
            
            # Content ID match
            if memory.content_id == context.get('content_id'):
                relevance += 0.5
            
            # Entry type relevance
            if memory.entry_type == context.get('query_type'):
                relevance += 0.3
            
            # Data similarity (simplified)
            memory_data = memory.data
            for key, value in context.items():
                if key in memory_data and memory_data[key] == value:
                    relevance += 0.1
            
            # Recency bonus
            age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
            recency_bonus = max(0, 0.2 * (1 - age_hours / (24 * 7)))  # Decay over a week
            relevance += recency_bonus
            
            # Importance factor
            relevance *= memory.importance
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating memory relevance: {e}")
            return 0.0
    
    def _get_score_range(self, score: float) -> str:
        """Convert numerical score to range category"""
        if score >= 0.8:
            return 'high'
        elif score >= 0.6:
            return 'medium-high'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'low-medium'
        else:
            return 'low'
    
    def _get_time_period(self, hour: int) -> str:
        """Convert hour to time period"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _patterns_match(self, pattern_conditions: Dict, content_conditions: Dict) -> bool:
        """Check if pattern conditions match content conditions"""
        try:
            match_count = 0
            total_conditions = len(pattern_conditions)
            
            for key, value in pattern_conditions.items():
                if key in content_conditions and content_conditions[key] == value:
                    match_count += 1
            
            # Require at least 70% match
            return (match_count / total_conditions) >= 0.7 if total_conditions > 0 else False
            
        except Exception as e:
            logger.error(f"Error matching patterns: {e}")
            return False
    
    def _consolidate_preferences(self):
        """Consolidate similar preferences to reduce redundancy"""
        try:
            # Group preferences by category
            category_groups = defaultdict(list)
            for pref in self.user_preferences.values():
                category_groups[pref.category].append(pref)
            
            # Consolidate within each category
            for category, prefs in category_groups.items():
                if len(prefs) > 5:  # Only consolidate if many preferences
                    # Keep only the most confident preferences
                    prefs.sort(key=lambda x: x.confidence, reverse=True)
                    to_keep = prefs[:3]  # Keep top 3
                    
                    # Remove others
                    for pref in prefs[3:]:
                        if pref.preference_id in self.user_preferences:
                            del self.user_preferences[pref.preference_id]
                            self._remove_preference_from_db(pref.preference_id)
            
        except Exception as e:
            logger.error(f"Error consolidating preferences: {e}")
    
    def _save_memory_entry(self, entry: MemoryEntry):
        """Save memory entry to database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memory_entries 
                (entry_id, entry_type, content_id, data, timestamp, importance, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.entry_id,
                entry.entry_type,
                entry.content_id,
                json.dumps(entry.data),
                entry.timestamp.isoformat(),
                entry.importance,
                entry.access_count,
                entry.last_accessed.isoformat() if entry.last_accessed else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving memory entry: {e}")
    
    def _save_user_preference(self, pref: UserPreference):
        """Save user preference to database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences 
                (preference_id, category, preference_type, value, confidence, evidence_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pref.preference_id,
                pref.category,
                pref.preference_type,
                json.dumps(pref.value),
                pref.confidence,
                pref.evidence_count,
                pref.last_updated.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving user preference: {e}")
    
    def _save_learning_pattern(self, pattern: LearningPattern):
        """Save learning pattern to database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_patterns 
                (pattern_id, pattern_type, description, conditions, outcomes, confidence, usage_count, success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                pattern.pattern_type,
                pattern.description,
                json.dumps(pattern.conditions),
                json.dumps(pattern.outcomes),
                pattern.confidence,
                pattern.usage_count,
                pattern.success_rate
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving learning pattern: {e}")
    
    def _remove_memory_from_db(self, entry_id: str):
        """Remove memory entry from database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_entries WHERE entry_id = ?", (entry_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error removing memory from database: {e}")
    
    def _remove_preference_from_db(self, preference_id: str):
        """Remove preference from database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_preferences WHERE preference_id = ?", (preference_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error removing preference from database: {e}")
    
    def export_memory_summary(self, filepath: str = "output/memory_summary.json"):
        """Export summary of agent's memory and learning"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            summary = {
                'memory_stats': {
                    'total_memories': len(self.memory_entries),
                    'total_preferences': len(self.user_preferences),
                    'total_patterns': len(self.learning_patterns),
                    'short_term_memory_size': len(self.short_term_memory)
                },
                'top_preferences': [
                    {
                        'category': pref.category,
                        'type': pref.preference_type,
                        'value': pref.value,
                        'confidence': pref.confidence
                    }
                    for pref in sorted(self.user_preferences.values(), 
                                     key=lambda x: x.confidence, reverse=True)[:10]
                ],
                'successful_patterns': [
                    {
                        'description': pattern.description,
                        'success_rate': pattern.success_rate,
                        'usage_count': pattern.usage_count,
                        'confidence': pattern.confidence
                    }
                    for pattern in sorted(self.learning_patterns.values(), 
                                        key=lambda x: x.success_rate, reverse=True)[:10]
                ],
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Memory summary exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting memory summary: {e}")
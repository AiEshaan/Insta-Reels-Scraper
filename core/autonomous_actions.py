"""
Autonomous Actions System
Provides intelligent, autonomous decision-making for content management
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of autonomous actions"""
    FILTER = "filter"
    CATEGORIZE = "categorize"
    SCHEDULE = "schedule"
    PRIORITIZE = "prioritize"
    ARCHIVE = "archive"
    ANALYZE_FURTHER = "analyze_further"
    RECOMMEND = "recommend"
    ALERT = "alert"

class Priority(Enum):
    """Priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class AutonomousAction:
    """Represents an autonomous action taken by the agent"""
    action_id: str
    action_type: ActionType
    target_content: str
    reasoning: str
    confidence: float
    priority: Priority
    parameters: Dict[str, Any]
    timestamp: datetime
    executed: bool = False
    result: Optional[str] = None

@dataclass
class FilterCriteria:
    """Criteria for content filtering"""
    min_quality_score: float = 0.3
    min_engagement_prediction: float = 0.4
    required_categories: List[str] = None
    excluded_categories: List[str] = None
    sentiment_filter: List[str] = None  # ['positive', 'neutral', 'negative']
    min_viral_potential: float = 0.2
    max_age_days: int = 30

@dataclass
class SchedulingRule:
    """Rules for autonomous scheduling"""
    optimal_hours: List[int] = None  # Hours of day (0-23)
    preferred_days: List[str] = None  # ['monday', 'tuesday', etc.]
    content_spacing_hours: int = 4
    max_daily_posts: int = 3
    category_rotation: bool = True

class AutonomousActionEngine:
    """
    Autonomous Action Engine for intelligent content management
    
    Features:
    - Smart content filtering based on quality and relevance
    - Intelligent categorization and tagging
    - Autonomous scheduling optimization
    - Priority-based content management
    - Learning from user feedback and patterns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.actions_history = []
        self.filter_criteria = FilterCriteria()
        self.scheduling_rules = SchedulingRule()
        self.user_preferences = {}
        self.performance_metrics = {}
        
        # Load configuration
        self._load_configuration()
        
        logger.info("Autonomous Action Engine initialized")
    
    def _load_configuration(self):
        """Load configuration from config dict or defaults"""
        # Filter criteria
        filter_config = self.config.get('filter_criteria', {})
        self.filter_criteria = FilterCriteria(
            min_quality_score=filter_config.get('min_quality_score', 0.3),
            min_engagement_prediction=filter_config.get('min_engagement_prediction', 0.4),
            required_categories=filter_config.get('required_categories'),
            excluded_categories=filter_config.get('excluded_categories'),
            sentiment_filter=filter_config.get('sentiment_filter'),
            min_viral_potential=filter_config.get('min_viral_potential', 0.2),
            max_age_days=filter_config.get('max_age_days', 30)
        )
        
        # Scheduling rules
        schedule_config = self.config.get('scheduling_rules', {})
        self.scheduling_rules = SchedulingRule(
            optimal_hours=schedule_config.get('optimal_hours', [9, 12, 15, 18, 21]),
            preferred_days=schedule_config.get('preferred_days'),
            content_spacing_hours=schedule_config.get('content_spacing_hours', 4),
            max_daily_posts=schedule_config.get('max_daily_posts', 3),
            category_rotation=schedule_config.get('category_rotation', True)
        )
    
    def execute_autonomous_pipeline(self, content_batch: List[Dict], analysis_results: Dict) -> Dict[str, Any]:
        """
        Execute the complete autonomous action pipeline
        """
        try:
            pipeline_results = {
                'total_content': len(content_batch),
                'actions_taken': [],
                'filtered_content': [],
                'categorized_content': [],
                'scheduled_content': [],
                'recommendations': [],
                'performance_metrics': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 1: Intelligent Filtering
            filtered_content = self._autonomous_filter(content_batch, analysis_results)
            pipeline_results['filtered_content'] = filtered_content
            
            # Step 2: Smart Categorization
            categorized_content = self._autonomous_categorize(filtered_content, analysis_results)
            pipeline_results['categorized_content'] = categorized_content
            
            # Step 3: Intelligent Scheduling
            scheduled_content = self._autonomous_schedule(categorized_content, analysis_results)
            pipeline_results['scheduled_content'] = scheduled_content
            
            # Step 4: Priority Assignment
            prioritized_content = self._autonomous_prioritize(scheduled_content, analysis_results)
            
            # Step 5: Generate Recommendations
            recommendations = self._generate_autonomous_recommendations(
                content_batch, filtered_content, analysis_results
            )
            pipeline_results['recommendations'] = recommendations
            
            # Step 6: Performance Analysis
            performance = self._analyze_pipeline_performance(pipeline_results)
            pipeline_results['performance_metrics'] = performance
            
            # Log all actions
            pipeline_results['actions_taken'] = [asdict(action) for action in self.actions_history[-10:]]
            
            logger.info(f"Autonomous pipeline completed: {len(filtered_content)}/{len(content_batch)} content items processed")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Error in autonomous pipeline: {e}")
            return {'error': str(e)}
    
    def _autonomous_filter(self, content_batch: List[Dict], analysis_results: Dict) -> List[Dict]:
        """Intelligently filter content based on quality and criteria"""
        filtered_content = []
        
        content_metrics = analysis_results.get('content_metrics', [])
        categories = analysis_results.get('content_categories', {}).get('individual_categories', [])
        
        for i, content in enumerate(content_batch):
            try:
                # Get metrics for this content
                metrics = content_metrics[i] if i < len(content_metrics) else None
                category = categories[i] if i < len(categories) else 'general'
                
                # Apply filtering logic
                should_include = self._evaluate_content_inclusion(content, metrics, category)
                
                if should_include:
                    # Add filtering metadata
                    content_copy = content.copy()
                    content_copy['filter_score'] = self._calculate_filter_score(metrics)
                    content_copy['assigned_category'] = category
                    content_copy['filter_timestamp'] = datetime.now().isoformat()
                    
                    filtered_content.append(content_copy)
                    
                    # Log action
                    action = AutonomousAction(
                        action_id=f"filter_{i}_{datetime.now().timestamp()}",
                        action_type=ActionType.FILTER,
                        target_content=content.get('Reel URL', ''),
                        reasoning=f"Content passed filter criteria with score {self._calculate_filter_score(metrics):.2f}",
                        confidence=0.8,
                        priority=Priority.MEDIUM,
                        parameters={'filter_score': self._calculate_filter_score(metrics)},
                        timestamp=datetime.now(),
                        executed=True,
                        result="included"
                    )
                    self.actions_history.append(action)
                else:
                    # Log rejection
                    action = AutonomousAction(
                        action_id=f"filter_reject_{i}_{datetime.now().timestamp()}",
                        action_type=ActionType.FILTER,
                        target_content=content.get('Reel URL', ''),
                        reasoning="Content did not meet filter criteria",
                        confidence=0.9,
                        priority=Priority.LOW,
                        parameters={'reason': 'quality_threshold'},
                        timestamp=datetime.now(),
                        executed=True,
                        result="excluded"
                    )
                    self.actions_history.append(action)
                    
            except Exception as e:
                logger.error(f"Error filtering content {i}: {e}")
                continue
        
        return filtered_content
    
    def _autonomous_categorize(self, content_batch: List[Dict], analysis_results: Dict) -> List[Dict]:
        """Intelligently categorize and tag content"""
        categorized_content = []
        
        for i, content in enumerate(content_batch):
            try:
                # Enhanced categorization
                category = content.get('assigned_category', 'general')
                caption = content.get('Caption', '')
                
                # Generate smart tags
                smart_tags = self._generate_smart_tags(caption, category)
                
                # Assign subcategory
                subcategory = self._assign_subcategory(caption, category)
                
                # Add categorization metadata
                content_copy = content.copy()
                content_copy.update({
                    'primary_category': category,
                    'subcategory': subcategory,
                    'smart_tags': smart_tags,
                    'categorization_confidence': self._calculate_categorization_confidence(caption, category),
                    'categorization_timestamp': datetime.now().isoformat()
                })
                
                categorized_content.append(content_copy)
                
                # Log action
                action = AutonomousAction(
                    action_id=f"categorize_{i}_{datetime.now().timestamp()}",
                    action_type=ActionType.CATEGORIZE,
                    target_content=content.get('Reel URL', ''),
                    reasoning=f"Categorized as {category}/{subcategory} with {len(smart_tags)} tags",
                    confidence=content_copy['categorization_confidence'],
                    priority=Priority.MEDIUM,
                    parameters={
                        'category': category,
                        'subcategory': subcategory,
                        'tags': smart_tags
                    },
                    timestamp=datetime.now(),
                    executed=True,
                    result="categorized"
                )
                self.actions_history.append(action)
                
            except Exception as e:
                logger.error(f"Error categorizing content {i}: {e}")
                categorized_content.append(content)
                continue
        
        return categorized_content
    
    def _autonomous_schedule(self, content_batch: List[Dict], analysis_results: Dict) -> List[Dict]:
        """Intelligently schedule content for optimal engagement"""
        scheduled_content = []
        
        # Sort content by priority/quality for scheduling
        sorted_content = sorted(
            content_batch,
            key=lambda x: x.get('filter_score', 0),
            reverse=True
        )
        
        current_time = datetime.now()
        scheduled_times = []
        
        for i, content in enumerate(sorted_content):
            try:
                # Calculate optimal posting time
                optimal_time = self._calculate_optimal_posting_time(
                    content, current_time, scheduled_times
                )
                
                # Add scheduling metadata
                content_copy = content.copy()
                content_copy.update({
                    'scheduled_time': optimal_time.isoformat(),
                    'scheduling_priority': i + 1,
                    'scheduling_reasoning': self._get_scheduling_reasoning(content, optimal_time),
                    'scheduling_confidence': self._calculate_scheduling_confidence(content, optimal_time),
                    'scheduling_timestamp': datetime.now().isoformat()
                })
                
                scheduled_content.append(content_copy)
                scheduled_times.append(optimal_time)
                
                # Log action
                action = AutonomousAction(
                    action_id=f"schedule_{i}_{datetime.now().timestamp()}",
                    action_type=ActionType.SCHEDULE,
                    target_content=content.get('Reel URL', ''),
                    reasoning=f"Scheduled for {optimal_time.strftime('%Y-%m-%d %H:%M')}",
                    confidence=content_copy['scheduling_confidence'],
                    priority=Priority.HIGH,
                    parameters={
                        'scheduled_time': optimal_time.isoformat(),
                        'priority': i + 1
                    },
                    timestamp=datetime.now(),
                    executed=True,
                    result="scheduled"
                )
                self.actions_history.append(action)
                
            except Exception as e:
                logger.error(f"Error scheduling content {i}: {e}")
                scheduled_content.append(content)
                continue
        
        return scheduled_content
    
    def _autonomous_prioritize(self, content_batch: List[Dict], analysis_results: Dict) -> List[Dict]:
        """Assign intelligent priorities to content"""
        for i, content in enumerate(content_batch):
            try:
                priority = self._calculate_content_priority(content, analysis_results)
                content['autonomous_priority'] = priority.value
                content['priority_reasoning'] = self._get_priority_reasoning(content, priority)
                
                # Log action
                action = AutonomousAction(
                    action_id=f"prioritize_{i}_{datetime.now().timestamp()}",
                    action_type=ActionType.PRIORITIZE,
                    target_content=content.get('Reel URL', ''),
                    reasoning=f"Assigned {priority.value} priority",
                    confidence=0.8,
                    priority=priority,
                    parameters={'priority': priority.value},
                    timestamp=datetime.now(),
                    executed=True,
                    result=priority.value
                )
                self.actions_history.append(action)
                
            except Exception as e:
                logger.error(f"Error prioritizing content {i}: {e}")
                continue
        
        return content_batch
    
    def _evaluate_content_inclusion(self, content: Dict, metrics: Any, category: str) -> bool:
        """Evaluate whether content should be included based on criteria"""
        if not metrics:
            return True  # Include if no metrics available
        
        # Quality score check
        if hasattr(metrics, 'engagement_prediction'):
            if metrics.engagement_prediction < self.filter_criteria.min_engagement_prediction:
                return False
        
        # Category filters
        if self.filter_criteria.required_categories:
            if category not in self.filter_criteria.required_categories:
                return False
        
        if self.filter_criteria.excluded_categories:
            if category in self.filter_criteria.excluded_categories:
                return False
        
        # Viral potential check
        if hasattr(metrics, 'viral_potential'):
            if metrics.viral_potential < self.filter_criteria.min_viral_potential:
                return False
        
        return True
    
    def _calculate_filter_score(self, metrics: Any) -> float:
        """Calculate a comprehensive filter score"""
        if not metrics:
            return 0.5
        
        score = 0.0
        if hasattr(metrics, 'engagement_prediction'):
            score += metrics.engagement_prediction * 0.4
        if hasattr(metrics, 'viral_potential'):
            score += metrics.viral_potential * 0.3
        if hasattr(metrics, 'authenticity_score'):
            score += metrics.authenticity_score * 0.2
        if hasattr(metrics, 'readability_score'):
            score += metrics.readability_score * 0.1
        
        return min(score, 1.0)
    
    def _generate_smart_tags(self, caption: str, category: str) -> List[str]:
        """Generate intelligent tags for content"""
        tags = [category]
        
        if not caption:
            return tags
        
        caption_lower = caption.lower()
        
        # Add emotion-based tags
        emotion_keywords = {
            'motivational': ['motivation', 'inspire', 'success', 'achieve', 'goal'],
            'educational': ['learn', 'tutorial', 'how-to', 'tips', 'guide'],
            'entertaining': ['fun', 'funny', 'hilarious', 'comedy', 'laugh'],
            'trending': ['viral', 'trending', 'popular', 'hot', 'buzz']
        }
        
        for tag, keywords in emotion_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                tags.append(tag)
        
        # Add time-based tags
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            tags.append('morning')
        elif 12 <= current_hour < 17:
            tags.append('afternoon')
        elif 17 <= current_hour < 21:
            tags.append('evening')
        else:
            tags.append('night')
        
        return list(set(tags))  # Remove duplicates
    
    def _assign_subcategory(self, caption: str, category: str) -> str:
        """Assign subcategory based on content analysis"""
        subcategories = {
            'lifestyle': ['daily_routine', 'home_decor', 'personal_growth', 'relationships'],
            'fitness': ['workout', 'nutrition', 'wellness', 'sports'],
            'food': ['recipes', 'restaurants', 'cooking_tips', 'food_reviews'],
            'travel': ['destinations', 'travel_tips', 'adventures', 'culture'],
            'fashion': ['outfits', 'trends', 'styling_tips', 'accessories'],
            'technology': ['gadgets', 'apps', 'tutorials', 'reviews'],
            'entertainment': ['movies', 'music', 'celebrities', 'events'],
            'education': ['tutorials', 'facts', 'skills', 'knowledge'],
            'business': ['entrepreneurship', 'marketing', 'productivity', 'success'],
            'art': ['photography', 'design', 'creativity', 'inspiration']
        }
        
        if category not in subcategories:
            return 'general'
        
        # Simple keyword matching for subcategory
        caption_lower = caption.lower() if caption else ''
        for subcategory in subcategories[category]:
            if subcategory.replace('_', ' ') in caption_lower:
                return subcategory
        
        return subcategories[category][0]  # Default to first subcategory
    
    def _calculate_categorization_confidence(self, caption: str, category: str) -> float:
        """Calculate confidence in categorization"""
        if not caption:
            return 0.5
        
        # Simple confidence based on keyword presence
        category_keywords = {
            'lifestyle': ['life', 'daily', 'routine', 'home', 'family'],
            'fitness': ['workout', 'gym', 'fitness', 'health', 'exercise'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal'],
            # Add more as needed
        }
        
        if category not in category_keywords:
            return 0.6
        
        caption_lower = caption.lower()
        keyword_matches = sum(1 for keyword in category_keywords[category] if keyword in caption_lower)
        
        confidence = min(0.5 + (keyword_matches * 0.1), 1.0)
        return confidence
    
    def _calculate_optimal_posting_time(self, content: Dict, current_time: datetime, 
                                      scheduled_times: List[datetime]) -> datetime:
        """Calculate optimal posting time for content"""
        # Start with next optimal hour
        optimal_hours = self.scheduling_rules.optimal_hours or [9, 12, 15, 18, 21]
        
        # Find next optimal hour
        next_optimal = current_time.replace(minute=0, second=0, microsecond=0)
        while next_optimal.hour not in optimal_hours:
            next_optimal += timedelta(hours=1)
        
        # Ensure spacing between posts
        while any(abs((next_optimal - scheduled).total_seconds()) < 
                 self.scheduling_rules.content_spacing_hours * 3600 
                 for scheduled in scheduled_times):
            next_optimal += timedelta(hours=1)
        
        # Check daily post limit
        same_day_posts = [t for t in scheduled_times if t.date() == next_optimal.date()]
        if len(same_day_posts) >= self.scheduling_rules.max_daily_posts:
            next_optimal = next_optimal.replace(hour=optimal_hours[0]) + timedelta(days=1)
        
        return next_optimal
    
    def _get_scheduling_reasoning(self, content: Dict, scheduled_time: datetime) -> str:
        """Get reasoning for scheduling decision"""
        hour = scheduled_time.hour
        
        if 6 <= hour < 12:
            return "Scheduled for morning engagement peak"
        elif 12 <= hour < 17:
            return "Scheduled for afternoon activity period"
        elif 17 <= hour < 21:
            return "Scheduled for evening prime time"
        else:
            return "Scheduled for optimal engagement window"
    
    def _calculate_scheduling_confidence(self, content: Dict, scheduled_time: datetime) -> float:
        """Calculate confidence in scheduling decision"""
        # Higher confidence for optimal hours
        optimal_hours = self.scheduling_rules.optimal_hours or [9, 12, 15, 18, 21]
        
        if scheduled_time.hour in optimal_hours:
            return 0.9
        elif abs(scheduled_time.hour - min(optimal_hours, key=lambda x: abs(x - scheduled_time.hour))) <= 1:
            return 0.7
        else:
            return 0.5
    
    def _calculate_content_priority(self, content: Dict, analysis_results: Dict) -> Priority:
        """Calculate priority for content"""
        filter_score = content.get('filter_score', 0.5)
        
        if filter_score >= 0.8:
            return Priority.CRITICAL
        elif filter_score >= 0.6:
            return Priority.HIGH
        elif filter_score >= 0.4:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _get_priority_reasoning(self, content: Dict, priority: Priority) -> str:
        """Get reasoning for priority assignment"""
        filter_score = content.get('filter_score', 0.5)
        
        return f"Priority {priority.value} assigned based on filter score {filter_score:.2f}"
    
    def _generate_autonomous_recommendations(self, original_content: List[Dict], 
                                           filtered_content: List[Dict], 
                                           analysis_results: Dict) -> List[str]:
        """Generate autonomous recommendations"""
        recommendations = []
        
        # Filter efficiency
        filter_rate = len(filtered_content) / len(original_content) if original_content else 0
        if filter_rate < 0.3:
            recommendations.append("Filter criteria may be too strict - consider adjusting thresholds")
        elif filter_rate > 0.9:
            recommendations.append("Filter criteria may be too lenient - consider raising quality standards")
        
        # Content diversity
        categories = [c.get('primary_category', 'general') for c in filtered_content]
        unique_categories = len(set(categories))
        if unique_categories < 3:
            recommendations.append("Content lacks diversity - consider expanding category coverage")
        
        # Scheduling optimization
        if len(filtered_content) > 10:
            recommendations.append("Large content batch detected - consider spreading posts over multiple days")
        
        return recommendations
    
    def _analyze_pipeline_performance(self, pipeline_results: Dict) -> Dict[str, Any]:
        """Analyze performance of the autonomous pipeline"""
        total_content = pipeline_results['total_content']
        filtered_count = len(pipeline_results['filtered_content'])
        
        performance = {
            'filter_efficiency': filtered_count / total_content if total_content > 0 else 0,
            'processing_success_rate': 1.0,  # Would be calculated based on errors
            'action_count': len(self.actions_history),
            'average_confidence': np.mean([a.confidence for a in self.actions_history[-10:]]) if self.actions_history else 0,
            'recommendation_count': len(pipeline_results['recommendations'])
        }
        
        return performance
    
    def export_actions_log(self, filepath: str = "output/autonomous_actions_log.json"):
        """Export log of all autonomous actions"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            actions_data = {
                'actions': [asdict(action) for action in self.actions_history],
                'total_actions': len(self.actions_history),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(actions_data, f, indent=2, default=str)
            
            logger.info(f"Actions log exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting actions log: {e}")
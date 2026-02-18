"""
Advanced Content Analysis Module
Provides intelligent content understanding, trend detection, and categorization
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

logger = logging.getLogger(__name__)

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    trending_topics: List[str]
    emerging_hashtags: List[str]
    sentiment_trends: Dict[str, float]
    engagement_patterns: Dict[str, Any]
    content_categories: List[str]
    trend_score: float
    analysis_timestamp: datetime

@dataclass
class ContentMetrics:
    """Comprehensive content metrics"""
    readability_score: float
    emotion_scores: Dict[str, float]
    topic_relevance: float
    viral_potential: float
    authenticity_score: float
    engagement_prediction: float

class AdvancedContentAnalyzer:
    """
    Advanced AI-powered content analysis system
    Features:
    - Sentiment and emotion analysis
    - Trend detection and prediction
    - Content categorization and clustering
    - Viral potential assessment
    - Engagement prediction
    - Topic modeling and relevance scoring
    """
    
    def __init__(self):
        # Initialize models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Content categories
        self.content_categories = {
            'lifestyle': ['life', 'daily', 'routine', 'home', 'family', 'personal'],
            'fitness': ['workout', 'gym', 'fitness', 'health', 'exercise', 'training'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal', 'delicious'],
            'travel': ['travel', 'vacation', 'trip', 'adventure', 'explore', 'destination'],
            'fashion': ['fashion', 'style', 'outfit', 'clothing', 'trend', 'look'],
            'technology': ['tech', 'gadget', 'app', 'digital', 'innovation', 'ai'],
            'entertainment': ['music', 'movie', 'show', 'celebrity', 'entertainment', 'fun'],
            'education': ['learn', 'tutorial', 'tips', 'how-to', 'guide', 'knowledge'],
            'business': ['business', 'entrepreneur', 'startup', 'marketing', 'success'],
            'art': ['art', 'creative', 'design', 'photography', 'artistic', 'aesthetic']
        }
        
        # Emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'amazing', 'wonderful', 'fantastic', 'love', 'awesome'],
            'sadness': ['sad', 'disappointed', 'upset', 'down', 'depressed', 'crying'],
            'anger': ['angry', 'mad', 'frustrated', 'annoyed', 'furious', 'hate'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
            'surprise': ['surprised', 'shocked', 'unexpected', 'wow', 'omg', 'unbelievable'],
            'trust': ['trust', 'reliable', 'honest', 'authentic', 'genuine', 'real']
        }
        
        # Viral indicators
        self.viral_indicators = [
            'viral', 'trending', 'challenge', 'reaction', 'shocking', 'unbelievable',
            'must-see', 'incredible', 'amazing', 'mind-blowing', 'epic', 'insane'
        ]
        
        logger.info("Advanced Content Analyzer initialized")
    
    def analyze_content_comprehensive(self, content_data: List[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a batch of content
        """
        try:
            captions = [item.get('Caption', '') for item in content_data]
            urls = [item.get('Reel URL', '') for item in content_data]
            
            # Perform various analyses
            sentiment_analysis = self._analyze_sentiment_batch(captions)
            trend_analysis = self._detect_trends(captions)
            categories = self._categorize_content(captions)
            metrics = self._calculate_content_metrics(captions)
            clusters = self._cluster_content(captions)
            
            analysis_results = {
                'total_content': len(content_data),
                'sentiment_distribution': sentiment_analysis,
                'trend_analysis': trend_analysis,
                'content_categories': categories,
                'content_metrics': metrics,
                'content_clusters': clusters,
                'recommendations': self._generate_recommendations(
                    sentiment_analysis, trend_analysis, categories, metrics
                ),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Comprehensive analysis completed for {len(content_data)} items")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {}
    
    def _analyze_sentiment_batch(self, captions: List[str]) -> Dict[str, Any]:
        """Analyze sentiment for batch of captions"""
        sentiments = []
        emotions = defaultdict(list)
        
        for caption in captions:
            if not caption:
                continue
                
            # TextBlob sentiment
            blob = TextBlob(caption)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            sentiments.append(sentiment)
            
            # Emotion analysis
            caption_lower = caption.lower()
            for emotion, keywords in self.emotion_keywords.items():
                emotion_score = sum(1 for keyword in keywords if keyword in caption_lower)
                emotions[emotion].append(emotion_score)
        
        sentiment_counts = Counter(sentiments)
        emotion_averages = {emotion: np.mean(scores) for emotion, scores in emotions.items()}
        
        return {
            'sentiment_distribution': dict(sentiment_counts),
            'emotion_scores': emotion_averages,
            'overall_sentiment': max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else 'neutral',
            'sentiment_confidence': max(sentiment_counts.values()) / len(sentiments) if sentiments else 0
        }
    
    def _detect_trends(self, captions: List[str]) -> TrendAnalysis:
        """Detect trends and patterns in content"""
        try:
            # Extract hashtags
            all_hashtags = []
            all_words = []
            
            for caption in captions:
                if not caption:
                    continue
                    
                # Extract hashtags
                hashtags = re.findall(r'#\w+', caption.lower())
                all_hashtags.extend(hashtags)
                
                # Extract words
                words = re.findall(r'\b\w+\b', caption.lower())
                all_words.extend([w for w in words if len(w) > 3])
            
            # Count frequencies
            hashtag_counts = Counter(all_hashtags)
            word_counts = Counter(all_words)
            
            # Identify trending topics
            trending_hashtags = [tag for tag, count in hashtag_counts.most_common(10)]
            trending_words = [word for word, count in word_counts.most_common(20)]
            
            # Calculate trend score
            trend_score = self._calculate_trend_score(captions)
            
            return TrendAnalysis(
                trending_topics=trending_words[:10],
                emerging_hashtags=trending_hashtags,
                sentiment_trends={},  # Would be calculated with historical data
                engagement_patterns={},  # Would be calculated with engagement data
                content_categories=list(self.content_categories.keys()),
                trend_score=trend_score,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in trend detection: {e}")
            return TrendAnalysis([], [], {}, {}, [], 0.0, datetime.now())
    
    def _categorize_content(self, captions: List[str]) -> Dict[str, Any]:
        """Categorize content into predefined categories"""
        category_scores = defaultdict(int)
        content_categories = []
        
        for caption in captions:
            if not caption:
                continue
                
            caption_lower = caption.lower()
            content_category_scores = {}
            
            for category, keywords in self.content_categories.items():
                score = sum(1 for keyword in keywords if keyword in caption_lower)
                content_category_scores[category] = score
                category_scores[category] += score
            
            # Assign primary category
            if content_category_scores:
                primary_category = max(content_category_scores, key=content_category_scores.get)
                if content_category_scores[primary_category] > 0:
                    content_categories.append(primary_category)
                else:
                    content_categories.append('general')
            else:
                content_categories.append('general')
        
        # Calculate category distribution
        category_distribution = Counter(content_categories)
        
        return {
            'category_distribution': dict(category_distribution),
            'dominant_category': max(category_distribution, key=category_distribution.get) if category_distribution else 'general',
            'category_diversity': len(category_distribution) / len(self.content_categories),
            'individual_categories': content_categories
        }
    
    def _calculate_content_metrics(self, captions: List[str]) -> List[ContentMetrics]:
        """Calculate detailed metrics for each piece of content"""
        metrics = []
        
        for caption in captions:
            if not caption:
                metrics.append(ContentMetrics(0, {}, 0, 0, 0, 0))
                continue
            
            # Readability score (simple version)
            words = caption.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            readability = max(0, 1 - (avg_word_length - 5) / 10)  # Normalize around 5-letter words
            
            # Emotion scores
            caption_lower = caption.lower()
            emotion_scores = {}
            for emotion, keywords in self.emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in caption_lower) / len(keywords)
                emotion_scores[emotion] = min(score, 1.0)
            
            # Viral potential
            viral_score = sum(1 for indicator in self.viral_indicators if indicator in caption_lower)
            viral_potential = min(viral_score / 3, 1.0)  # Normalize
            
            # Authenticity score (based on personal pronouns and authentic language)
            personal_pronouns = ['i', 'me', 'my', 'we', 'us', 'our']
            authenticity = sum(1 for pronoun in personal_pronouns if pronoun in caption_lower.split())
            authenticity_score = min(authenticity / 5, 1.0)
            
            # Topic relevance (placeholder - would use more sophisticated NLP)
            topic_relevance = 0.5  # Default
            
            # Engagement prediction (based on various factors)
            engagement_prediction = (
                viral_potential * 0.3 +
                emotion_scores.get('joy', 0) * 0.2 +
                authenticity_score * 0.2 +
                readability * 0.15 +
                topic_relevance * 0.15
            )
            
            metrics.append(ContentMetrics(
                readability_score=readability,
                emotion_scores=emotion_scores,
                topic_relevance=topic_relevance,
                viral_potential=viral_potential,
                authenticity_score=authenticity_score,
                engagement_prediction=engagement_prediction
            ))
        
        return metrics
    
    def _cluster_content(self, captions: List[str]) -> Dict[str, Any]:
        """Cluster content using embeddings"""
        try:
            if len(captions) < 3:
                return {'clusters': [], 'cluster_labels': []}
            
            # Filter out empty captions
            valid_captions = [c for c in captions if c.strip()]
            if len(valid_captions) < 3:
                return {'clusters': [], 'cluster_labels': []}
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(valid_captions)
            
            # Determine optimal number of clusters
            n_clusters = min(5, len(valid_captions) // 2)
            if n_clusters < 2:
                n_clusters = 2
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group content by clusters
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[f"cluster_{label}"].append(valid_captions[i])
            
            return {
                'clusters': dict(clusters),
                'cluster_labels': cluster_labels.tolist(),
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error in content clustering: {e}")
            return {'clusters': {}, 'cluster_labels': []}
    
    def _calculate_trend_score(self, captions: List[str]) -> float:
        """Calculate overall trend score for content batch"""
        try:
            trend_indicators = 0
            total_content = len(captions)
            
            for caption in captions:
                if not caption:
                    continue
                    
                caption_lower = caption.lower()
                
                # Check for viral indicators
                if any(indicator in caption_lower for indicator in self.viral_indicators):
                    trend_indicators += 1
                
                # Check for hashtags (trending indicator)
                if '#' in caption:
                    trend_indicators += 0.5
                
                # Check for current events keywords (would be more sophisticated in practice)
                current_keywords = ['2024', 'new', 'latest', 'trending', 'viral']
                if any(keyword in caption_lower for keyword in current_keywords):
                    trend_indicators += 0.3
            
            trend_score = min(trend_indicators / total_content, 1.0) if total_content > 0 else 0
            return trend_score
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0
    
    def _generate_recommendations(self, sentiment_analysis: Dict, trend_analysis: TrendAnalysis, 
                                categories: Dict, metrics: List[ContentMetrics]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Sentiment-based recommendations
        if sentiment_analysis.get('overall_sentiment') == 'negative':
            recommendations.append("Consider focusing on more positive content to improve engagement")
        
        # Trend-based recommendations
        if trend_analysis.trend_score > 0.7:
            recommendations.append("High trend potential detected - consider prioritizing this content")
        elif trend_analysis.trend_score < 0.3:
            recommendations.append("Low trend potential - consider adding trending hashtags or topics")
        
        # Category-based recommendations
        category_diversity = categories.get('category_diversity', 0)
        if category_diversity < 0.3:
            recommendations.append("Content lacks diversity - consider exploring different categories")
        
        # Metrics-based recommendations
        if metrics:
            avg_engagement = np.mean([m.engagement_prediction for m in metrics])
            if avg_engagement < 0.5:
                recommendations.append("Low engagement prediction - consider improving content quality and authenticity")
            
            avg_viral_potential = np.mean([m.viral_potential for m in metrics])
            if avg_viral_potential < 0.3:
                recommendations.append("Add more engaging elements to increase viral potential")
        
        return recommendations
    
    def export_analysis_report(self, analysis_results: Dict, filepath: str = "output/content_analysis_report.json"):
        """Export detailed analysis report"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Analysis report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
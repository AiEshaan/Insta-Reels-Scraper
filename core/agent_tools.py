"""
Agent Tools
Comprehensive toolkit for AI agent capabilities including web search, content generation, and API interactions
"""

import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os
import re

import requests
from bs4 import BeautifulSoup
import openai
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Web search result"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    timestamp: datetime

@dataclass
class ContentGenerationRequest:
    """Content generation request"""
    content_type: str  # 'caption', 'hashtags', 'description', 'summary'
    input_data: Dict[str, Any]
    style: str = 'engaging'
    max_length: int = 280
    target_audience: str = 'general'

@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    response_time: float = 0.0
    source: str = 'unknown'

class WebSearchTool:
    """
    Advanced Web Search Tool
    
    Features:
    - Multiple search engines (Google, Bing, DuckDuckGo)
    - Trend analysis and real-time information
    - Content verification and fact-checking
    - Social media trend detection
    """
    
    def __init__(self, google_api_key: str = None, google_cse_id: str = None):
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = google_cse_id or os.getenv('GOOGLE_CSE_ID')
        
        # Initialize search wrappers
        if self.google_api_key and self.google_cse_id:
            self.google_search = GoogleSearchAPIWrapper(
                google_api_key=self.google_api_key,
                google_cse_id=self.google_cse_id
            )
        else:
            self.google_search = None
            logger.warning("Google Search API not configured")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_trends(self, query: str, timeframe: str = 'week') -> List[SearchResult]:
        """Search for trending topics and information"""
        try:
            results = []
            
            # Google Trends search
            if self.google_search:
                google_results = self.google_search.results(f"{query} trends {timeframe}", num_results=5)
                for result in google_results:
                    results.append(SearchResult(
                        title=result.get('title', ''),
                        url=result.get('link', ''),
                        snippet=result.get('snippet', ''),
                        source='google',
                        relevance_score=0.8,
                        timestamp=datetime.now()
                    ))
            
            # DuckDuckGo fallback
            if not results:
                results = self._duckduckgo_search(f"{query} trends {timeframe}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching trends: {e}")
            return []
    
    def search_real_time(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for real-time information"""
        try:
            results = []
            
            # Add time-based modifiers
            time_query = f"{query} latest news today"
            
            if self.google_search:
                google_results = self.google_search.results(time_query, num_results=max_results)
                for result in google_results:
                    results.append(SearchResult(
                        title=result.get('title', ''),
                        url=result.get('link', ''),
                        snippet=result.get('snippet', ''),
                        source='google',
                        relevance_score=self._calculate_relevance(result, query),
                        timestamp=datetime.now()
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in real-time search: {e}")
            return []
    
    def verify_information(self, claim: str) -> Dict[str, Any]:
        """Verify information using multiple sources"""
        try:
            verification_results = {
                'claim': claim,
                'verification_score': 0.5,
                'sources': [],
                'confidence': 'medium',
                'fact_check_results': []
            }
            
            # Search for fact-checking sources
            fact_check_query = f'"{claim}" fact check verification'
            results = self.search_real_time(fact_check_query, max_results=5)
            
            # Analyze results for credibility
            credible_sources = ['snopes.com', 'factcheck.org', 'politifact.com', 'reuters.com', 'bbc.com']
            
            for result in results:
                source_credibility = 0.5
                for credible_source in credible_sources:
                    if credible_source in result.url.lower():
                        source_credibility = 0.9
                        break
                
                verification_results['sources'].append({
                    'title': result.title,
                    'url': result.url,
                    'credibility': source_credibility,
                    'snippet': result.snippet
                })
            
            # Calculate overall verification score
            if verification_results['sources']:
                avg_credibility = sum(s['credibility'] for s in verification_results['sources']) / len(verification_results['sources'])
                verification_results['verification_score'] = avg_credibility
                
                if avg_credibility > 0.8:
                    verification_results['confidence'] = 'high'
                elif avg_credibility > 0.6:
                    verification_results['confidence'] = 'medium'
                else:
                    verification_results['confidence'] = 'low'
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error verifying information: {e}")
            return {'claim': claim, 'verification_score': 0.5, 'sources': [], 'confidence': 'unknown'}
    
    def _duckduckgo_search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Fallback search using DuckDuckGo"""
        try:
            results = []
            
            # Simple DuckDuckGo search (no API key required)
            search_url = f"https://duckduckgo.com/html/?q={query}"
            response = self.session.get(search_url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                search_results = soup.find_all('div', class_='result')
                
                for i, result in enumerate(search_results[:max_results]):
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        results.append(SearchResult(
                            title=title_elem.get_text(strip=True),
                            url=title_elem.get('href', ''),
                            snippet=snippet_elem.get_text(strip=True),
                            source='duckduckgo',
                            relevance_score=max(0.1, 0.9 - i * 0.1),
                            timestamp=datetime.now()
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return []
    
    def _calculate_relevance(self, result: Dict, query: str) -> float:
        """Calculate relevance score for search result"""
        try:
            relevance = 0.0
            query_terms = query.lower().split()
            
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # Term matching in title (higher weight)
            title_matches = sum(1 for term in query_terms if term in title)
            relevance += (title_matches / len(query_terms)) * 0.6
            
            # Term matching in snippet
            snippet_matches = sum(1 for term in query_terms if term in snippet)
            relevance += (snippet_matches / len(query_terms)) * 0.4
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.5

class ContentGenerationTool:
    """
    Advanced Content Generation Tool
    
    Features:
    - Multi-format content generation (captions, hashtags, descriptions)
    - Style and tone adaptation
    - Audience-specific content
    - Template-based generation
    - Content optimization for engagement
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            logger.warning("OpenAI API key not configured")
        
        # Content templates
        self.templates = {
            'caption': {
                'engaging': "Create an engaging Instagram caption for {content_type} content about {topic}. Make it {tone} and include a call-to-action. Max {max_length} characters.",
                'professional': "Write a professional caption for {content_type} content about {topic}. Keep it {tone} and informative. Max {max_length} characters.",
                'casual': "Write a casual, friendly caption for {content_type} content about {topic}. Make it {tone} and relatable. Max {max_length} characters."
            },
            'hashtags': {
                'trending': "Generate 10-15 trending hashtags for {content_type} content about {topic}. Include a mix of popular and niche hashtags.",
                'niche': "Generate 10-15 niche-specific hashtags for {content_type} content about {topic}. Focus on targeted, specific hashtags.",
                'viral': "Generate 10-15 hashtags with viral potential for {content_type} content about {topic}. Include trending and engaging hashtags."
            },
            'description': {
                'detailed': "Write a detailed description for {content_type} content about {topic}. Include key points, benefits, and context. Max {max_length} words.",
                'summary': "Write a concise summary description for {content_type} content about {topic}. Highlight main points. Max {max_length} words.",
                'seo': "Write an SEO-optimized description for {content_type} content about {topic}. Include relevant keywords. Max {max_length} words."
            }
        }
    
    def generate_content(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate content based on request parameters"""
        try:
            if not self.openai_api_key:
                return self._generate_template_content(request)
            
            # Prepare prompt
            prompt = self._build_prompt(request)
            
            # Generate content using OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative content generator specializing in social media content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=min(request.max_length * 2, 500),
                temperature=0.7
            )
            
            generated_content = response.choices[0].message.content.strip()
            
            # Post-process content
            processed_content = self._post_process_content(generated_content, request)
            
            return {
                'success': True,
                'content': processed_content,
                'content_type': request.content_type,
                'style': request.style,
                'length': len(processed_content),
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return self._generate_template_content(request)
    
    def generate_multiple_variants(self, request: ContentGenerationRequest, count: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple content variants"""
        try:
            variants = []
            
            for i in range(count):
                # Vary the style slightly for each variant
                variant_request = ContentGenerationRequest(
                    content_type=request.content_type,
                    input_data=request.input_data,
                    style=request.style,
                    max_length=request.max_length,
                    target_audience=request.target_audience
                )
                
                variant = self.generate_content(variant_request)
                if variant['success']:
                    variant['variant_id'] = i + 1
                    variants.append(variant)
            
            return variants
            
        except Exception as e:
            logger.error(f"Error generating content variants: {e}")
            return []
    
    def optimize_for_engagement(self, content: str, content_type: str) -> Dict[str, Any]:
        """Optimize content for better engagement"""
        try:
            optimization_suggestions = {
                'original_content': content,
                'optimized_content': content,
                'suggestions': [],
                'engagement_score': 0.5
            }
            
            # Analyze content for optimization opportunities
            if content_type == 'caption':
                # Add emojis if missing
                if not any(char for char in content if ord(char) > 127):
                    optimization_suggestions['suggestions'].append("Add relevant emojis to increase visual appeal")
                
                # Check for call-to-action
                cta_keywords = ['comment', 'share', 'like', 'follow', 'tag', 'dm', 'swipe']
                if not any(keyword in content.lower() for keyword in cta_keywords):
                    optimization_suggestions['suggestions'].append("Add a call-to-action to encourage engagement")
                
                # Check length
                if len(content) > 200:
                    optimization_suggestions['suggestions'].append("Consider shortening for better readability")
                elif len(content) < 50:
                    optimization_suggestions['suggestions'].append("Consider adding more context or details")
            
            elif content_type == 'hashtags':
                hashtags = re.findall(r'#\w+', content)
                if len(hashtags) < 5:
                    optimization_suggestions['suggestions'].append("Add more hashtags for better discoverability")
                elif len(hashtags) > 20:
                    optimization_suggestions['suggestions'].append("Reduce hashtags to avoid appearing spammy")
            
            # Calculate engagement score based on factors
            engagement_score = self._calculate_engagement_score(content, content_type)
            optimization_suggestions['engagement_score'] = engagement_score
            
            return optimization_suggestions
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return {'original_content': content, 'optimized_content': content, 'suggestions': [], 'engagement_score': 0.5}
    
    def _build_prompt(self, request: ContentGenerationRequest) -> str:
        """Build prompt for content generation"""
        try:
            # Extract key information from input data
            topic = request.input_data.get('topic', 'general content')
            content_category = request.input_data.get('category', 'entertainment')
            tone = request.input_data.get('tone', 'friendly')
            
            # Get template
            template_category = request.content_type
            template_style = request.style if request.style in self.templates.get(template_category, {}) else 'engaging'
            
            if template_category in self.templates and template_style in self.templates[template_category]:
                prompt = self.templates[template_category][template_style].format(
                    content_type=content_category,
                    topic=topic,
                    tone=tone,
                    max_length=request.max_length
                )
            else:
                # Fallback prompt
                prompt = f"Create {request.content_type} content about {topic} for {request.target_audience} audience. Style: {request.style}. Max length: {request.max_length}."
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            return f"Create {request.content_type} content."
    
    def _post_process_content(self, content: str, request: ContentGenerationRequest) -> str:
        """Post-process generated content"""
        try:
            processed = content.strip()
            
            # Ensure length constraints
            if len(processed) > request.max_length:
                processed = processed[:request.max_length-3] + "..."
            
            # Content-type specific processing
            if request.content_type == 'hashtags':
                # Ensure hashtags start with #
                lines = processed.split('\n')
                hashtag_lines = []
                for line in lines:
                    if line.strip() and not line.strip().startswith('#'):
                        line = '#' + line.strip()
                    hashtag_lines.append(line)
                processed = '\n'.join(hashtag_lines)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error post-processing content: {e}")
            return content
    
    def _generate_template_content(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate content using templates (fallback)"""
        try:
            topic = request.input_data.get('topic', 'content')
            category = request.input_data.get('category', 'general')
            
            if request.content_type == 'caption':
                content = f"Check out this amazing {category} content about {topic}! What do you think? Let me know in the comments! 🔥 #content #{category.replace(' ', '')}"
            elif request.content_type == 'hashtags':
                content = f"#{category.replace(' ', '')} #{topic.replace(' ', '')} #viral #trending #content #instagram #explore #fyp #amazing #cool #awesome"
            elif request.content_type == 'description':
                content = f"This {category} content focuses on {topic}. It provides valuable insights and engaging material for viewers interested in this topic."
            else:
                content = f"Generated {request.content_type} content about {topic}"
            
            return {
                'success': True,
                'content': content[:request.max_length],
                'content_type': request.content_type,
                'style': request.style,
                'length': len(content[:request.max_length]),
                'generation_time': datetime.now().isoformat(),
                'method': 'template'
            }
            
        except Exception as e:
            logger.error(f"Error generating template content: {e}")
            return {
                'success': False,
                'content': '',
                'error': str(e)
            }
    
    def _calculate_engagement_score(self, content: str, content_type: str) -> float:
        """Calculate predicted engagement score"""
        try:
            score = 0.5  # Base score
            
            if content_type == 'caption':
                # Length factor
                if 50 <= len(content) <= 150:
                    score += 0.1
                
                # Emoji factor
                emoji_count = sum(1 for char in content if ord(char) > 127)
                if 1 <= emoji_count <= 5:
                    score += 0.1
                
                # Question factor
                if '?' in content:
                    score += 0.1
                
                # Call-to-action factor
                cta_keywords = ['comment', 'share', 'like', 'follow', 'tag']
                if any(keyword in content.lower() for keyword in cta_keywords):
                    score += 0.2
            
            elif content_type == 'hashtags':
                hashtag_count = len(re.findall(r'#\w+', content))
                if 5 <= hashtag_count <= 15:
                    score += 0.2
                elif hashtag_count > 15:
                    score -= 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.5

class ExternalAPITool:
    """
    External API Integration Tool
    
    Features:
    - Social media APIs (Instagram, Twitter, TikTok)
    - Analytics APIs
    - Content management APIs
    - Webhook handling
    - Rate limiting and error handling
    """
    
    def __init__(self):
        self.session = None
        self.rate_limits = {}
        self.api_keys = {
            'instagram': os.getenv('INSTAGRAM_API_KEY'),
            'twitter': os.getenv('TWITTER_API_KEY'),
            'youtube': os.getenv('YOUTUBE_API_KEY'),
            'tiktok': os.getenv('TIKTOK_API_KEY')
        }
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_social_media_trends(self, platform: str, category: str = 'general') -> APIResponse:
        """Get trending content from social media platforms"""
        try:
            start_time = datetime.now()
            
            if platform.lower() == 'instagram':
                data = await self._get_instagram_trends(category)
            elif platform.lower() == 'twitter':
                data = await self._get_twitter_trends(category)
            elif platform.lower() == 'youtube':
                data = await self._get_youtube_trends(category)
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    error_message=f"Platform {platform} not supported",
                    source=platform
                )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return APIResponse(
                success=True,
                data=data,
                response_time=response_time,
                source=platform
            )
            
        except Exception as e:
            logger.error(f"Error getting {platform} trends: {e}")
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                source=platform
            )
    
    async def get_content_analytics(self, platform: str, content_id: str) -> APIResponse:
        """Get analytics for specific content"""
        try:
            start_time = datetime.now()
            
            # Simulate analytics data (replace with actual API calls)
            analytics_data = {
                'content_id': content_id,
                'platform': platform,
                'views': 1250,
                'likes': 89,
                'shares': 23,
                'comments': 15,
                'engagement_rate': 0.102,
                'reach': 1890,
                'impressions': 2340,
                'timestamp': datetime.now().isoformat()
            }
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return APIResponse(
                success=True,
                data=analytics_data,
                response_time=response_time,
                source=f"{platform}_analytics"
            )
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                source=f"{platform}_analytics"
            )
    
    async def post_content(self, platform: str, content_data: Dict[str, Any]) -> APIResponse:
        """Post content to social media platform"""
        try:
            start_time = datetime.now()
            
            # Simulate posting (replace with actual API calls)
            post_result = {
                'post_id': f"{platform}_{datetime.now().timestamp()}",
                'platform': platform,
                'status': 'published',
                'url': f"https://{platform}.com/post/{datetime.now().timestamp()}",
                'scheduled_time': content_data.get('scheduled_time'),
                'timestamp': datetime.now().isoformat()
            }
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return APIResponse(
                success=True,
                data=post_result,
                response_time=response_time,
                source=f"{platform}_post"
            )
            
        except Exception as e:
            logger.error(f"Error posting to {platform}: {e}")
            return APIResponse(
                success=False,
                data=None,
                error_message=str(e),
                source=f"{platform}_post"
            )
    
    async def _get_instagram_trends(self, category: str) -> Dict[str, Any]:
        """Get Instagram trending content"""
        # Placeholder implementation
        return {
            'trends': [
                {'hashtag': '#trending', 'posts': 125000, 'growth': 15.2},
                {'hashtag': '#viral', 'posts': 89000, 'growth': 22.1},
                {'hashtag': f'#{category}', 'posts': 45000, 'growth': 8.7}
            ],
            'category': category,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_twitter_trends(self, category: str) -> Dict[str, Any]:
        """Get Twitter trending topics"""
        # Placeholder implementation
        return {
            'trends': [
                {'topic': 'Trending Topic 1', 'tweets': 25000, 'volume': 'high'},
                {'topic': 'Trending Topic 2', 'tweets': 18000, 'volume': 'medium'},
                {'topic': f'{category} trends', 'tweets': 12000, 'volume': 'medium'}
            ],
            'category': category,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_youtube_trends(self, category: str) -> Dict[str, Any]:
        """Get YouTube trending videos"""
        # Placeholder implementation
        return {
            'trends': [
                {'title': 'Trending Video 1', 'views': 1500000, 'category': category},
                {'title': 'Trending Video 2', 'views': 890000, 'category': category},
                {'title': 'Trending Video 3', 'views': 650000, 'category': category}
            ],
            'category': category,
            'timestamp': datetime.now().isoformat()
        }
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

class AgentToolsManager:
    """
    Central manager for all agent tools
    
    Coordinates between different tools and provides unified interface
    """
    
    def __init__(self):
        self.web_search = WebSearchTool()
        self.content_generator = ContentGenerationTool()
        self.api_tool = ExternalAPITool()
        
        logger.info("Agent Tools Manager initialized")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [
            'web_search',
            'content_generation',
            'trend_analysis',
            'information_verification',
            'social_media_integration',
            'analytics_tracking',
            'content_optimization'
        ]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool with given parameters"""
        try:
            if tool_name == 'web_search':
                query = kwargs.get('query', '')
                search_type = kwargs.get('search_type', 'general')
                
                if search_type == 'trends':
                    results = self.web_search.search_trends(query, kwargs.get('timeframe', 'week'))
                elif search_type == 'real_time':
                    results = self.web_search.search_real_time(query, kwargs.get('max_results', 10))
                else:
                    results = self.web_search.search_real_time(query, kwargs.get('max_results', 10))
                
                return {'success': True, 'results': [result.__dict__ for result in results]}
            
            elif tool_name == 'content_generation':
                request = ContentGenerationRequest(**kwargs)
                result = self.content_generator.generate_content(request)
                return result
            
            elif tool_name == 'information_verification':
                claim = kwargs.get('claim', '')
                result = self.web_search.verify_information(claim)
                return {'success': True, 'verification': result}
            
            elif tool_name == 'social_media_trends':
                platform = kwargs.get('platform', 'instagram')
                category = kwargs.get('category', 'general')
                result = await self.api_tool.get_social_media_trends(platform, category)
                return {'success': result.success, 'data': result.data, 'error': result.error_message}
            
            else:
                return {'success': False, 'error': f'Tool {tool_name} not found'}
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.api_tool.close()
        logger.info("Agent tools cleaned up")

# Tool factory for LangChain integration
def create_langchain_tools() -> List[Tool]:
    """Create LangChain-compatible tools"""
    tools_manager = AgentToolsManager()
    
    tools = [
        Tool(
            name="web_search",
            description="Search the web for information, trends, and real-time data",
            func=lambda query: asyncio.run(tools_manager.execute_tool('web_search', query=query))
        ),
        Tool(
            name="content_generation",
            description="Generate social media content including captions, hashtags, and descriptions",
            func=lambda content_type, topic: asyncio.run(tools_manager.execute_tool(
                'content_generation',
                content_type=content_type,
                input_data={'topic': topic}
            ))
        ),
        Tool(
            name="verify_information",
            description="Verify claims and information using multiple sources",
            func=lambda claim: asyncio.run(tools_manager.execute_tool('information_verification', claim=claim))
        )
    ]
    
    return tools
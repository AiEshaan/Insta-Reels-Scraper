"""
AI Agent Configuration
Centralized configuration for all AI agent components
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for Language Model"""
    provider: str = "openai"  # openai, anthropic, huggingface
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30

@dataclass
class VectorStoreConfig:
    """Configuration for Vector Store"""
    provider: str = "chromadb"  # chromadb, pinecone, weaviate
    collection_name: str = "instagram_reels"
    persist_directory: str = "data/vectorstore"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class MemoryConfig:
    """Configuration for Agent Memory"""
    database_path: str = "data/agent_memory.db"
    max_memories: int = 10000
    consolidation_threshold: int = 100
    importance_threshold: float = 0.5

@dataclass
class ContentAnalysisConfig:
    """Configuration for Content Analysis"""
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    viral_threshold: float = 0.7
    engagement_threshold: float = 0.6

@dataclass
class AutonomousActionsConfig:
    """Configuration for Autonomous Actions"""
    max_daily_actions: int = 100
    confidence_threshold: float = 0.7
    enable_filtering: bool = True
    enable_categorization: bool = True
    enable_scheduling: bool = True
    enable_recommendations: bool = True

@dataclass
class WebSearchConfig:
    """Configuration for Web Search Tools"""
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    bing_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    max_results: int = 10
    timeout: int = 30

@dataclass
class SocialMediaConfig:
    """Configuration for Social Media APIs"""
    instagram_access_token: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    facebook_access_token: Optional[str] = None
    tiktok_access_token: Optional[str] = None

class AIAgentConfig:
    """Main AI Agent Configuration"""
    
    def __init__(self):
        # Load environment variables
        self.load_from_env()
        
        # Initialize component configurations
        self.llm = LLMConfig(
            api_key=self.openai_api_key,
            model_name=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
        )
        
        self.vector_store = VectorStoreConfig(
            persist_directory=os.getenv('VECTORSTORE_PATH', 'data/vectorstore'),
            collection_name=os.getenv('VECTORSTORE_COLLECTION', 'instagram_reels')
        )
        
        self.memory = MemoryConfig(
            database_path=os.getenv('MEMORY_DB_PATH', 'data/agent_memory.db'),
            max_memories=int(os.getenv('MAX_MEMORIES', '10000'))
        )
        
        self.content_analysis = ContentAnalysisConfig(
            viral_threshold=float(os.getenv('VIRAL_THRESHOLD', '0.7')),
            engagement_threshold=float(os.getenv('ENGAGEMENT_THRESHOLD', '0.6'))
        )
        
        self.autonomous_actions = AutonomousActionsConfig(
            max_daily_actions=int(os.getenv('MAX_DAILY_ACTIONS', '100')),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')),
            enable_filtering=os.getenv('ENABLE_FILTERING', 'true').lower() == 'true',
            enable_categorization=os.getenv('ENABLE_CATEGORIZATION', 'true').lower() == 'true'
        )
        
        self.web_search = WebSearchConfig(
            google_api_key=self.google_api_key,
            google_cse_id=self.google_cse_id,
            bing_api_key=self.bing_api_key,
            serpapi_key=self.serpapi_key
        )
        
        self.social_media = SocialMediaConfig(
            instagram_access_token=self.instagram_access_token,
            twitter_api_key=self.twitter_api_key,
            twitter_api_secret=self.twitter_api_secret
        )
        
        # General settings
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.output_directory = os.getenv('OUTPUT_DIRECTORY', 'output')
        self.data_directory = os.getenv('DATA_DIRECTORY', 'data')
        
        # Create directories if they don't exist
        self.ensure_directories()
    
    def load_from_env(self):
        """Load API keys and sensitive data from environment variables"""
        # OpenAI
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Google Search
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        
        # Bing Search
        self.bing_api_key = os.getenv('BING_API_KEY')
        
        # SerpAPI
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        
        # Social Media APIs
        self.instagram_access_token = os.getenv('INSTAGRAM_ACCESS_TOKEN')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.facebook_access_token = os.getenv('FACEBOOK_ACCESS_TOKEN')
        self.tiktok_access_token = os.getenv('TIKTOK_ACCESS_TOKEN')
        
        # Anthropic
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Hugging Face
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.output_directory,
            self.data_directory,
            self.vector_store.persist_directory,
            os.path.dirname(self.memory.database_path),
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_api_status(self) -> Dict[str, bool]:
        """Check which APIs are configured"""
        return {
            'openai': bool(self.openai_api_key),
            'google_search': bool(self.google_api_key and self.google_cse_id),
            'bing_search': bool(self.bing_api_key),
            'serpapi': bool(self.serpapi_key),
            'instagram': bool(self.instagram_access_token),
            'twitter': bool(self.twitter_api_key and self.twitter_api_secret),
            'facebook': bool(self.facebook_access_token),
            'tiktok': bool(self.tiktok_access_token),
            'anthropic': bool(self.anthropic_api_key),
            'huggingface': bool(self.huggingface_api_key)
        }
    
    def get_missing_apis(self) -> List[str]:
        """Get list of missing API configurations"""
        status = self.get_api_status()
        return [api for api, configured in status.items() if not configured]
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check critical APIs
        if not self.openai_api_key:
            issues['errors'].append('OpenAI API key is required for LLM functionality')
        
        # Check optional but recommended APIs
        missing_apis = self.get_missing_apis()
        if 'google_search' in missing_apis and 'bing_search' in missing_apis and 'serpapi' in missing_apis:
            issues['warnings'].append('No web search API configured - web search features will be limited')
        
        if 'instagram' in missing_apis:
            issues['warnings'].append('Instagram API not configured - real-time data features will be limited')
        
        # Check thresholds
        if self.content_analysis.viral_threshold > 0.9:
            issues['warnings'].append('Viral threshold is very high - may miss potential viral content')
        
        if self.autonomous_actions.max_daily_actions < 10:
            issues['warnings'].append('Max daily actions is very low - agent may be too conservative')
        
        # Info messages
        if self.debug_mode:
            issues['info'].append('Debug mode is enabled')
        
        configured_apis = [api for api, configured in self.get_api_status().items() if configured]
        issues['info'].append(f'Configured APIs: {", ".join(configured_apis)}')
        
        return issues
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'llm': {
                'provider': self.llm.provider,
                'model_name': self.llm.model_name,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'api_key_configured': bool(self.llm.api_key)
            },
            'vector_store': {
                'provider': self.vector_store.provider,
                'collection_name': self.vector_store.collection_name,
                'persist_directory': self.vector_store.persist_directory
            },
            'memory': {
                'database_path': self.memory.database_path,
                'max_memories': self.memory.max_memories
            },
            'content_analysis': {
                'viral_threshold': self.content_analysis.viral_threshold,
                'engagement_threshold': self.content_analysis.engagement_threshold
            },
            'autonomous_actions': {
                'max_daily_actions': self.autonomous_actions.max_daily_actions,
                'confidence_threshold': self.autonomous_actions.confidence_threshold,
                'features_enabled': {
                    'filtering': self.autonomous_actions.enable_filtering,
                    'categorization': self.autonomous_actions.enable_categorization,
                    'scheduling': self.autonomous_actions.enable_scheduling,
                    'recommendations': self.autonomous_actions.enable_recommendations
                }
            },
            'api_status': self.get_api_status(),
            'general': {
                'debug_mode': self.debug_mode,
                'log_level': self.log_level,
                'output_directory': self.output_directory,
                'data_directory': self.data_directory
            }
        }

# Global configuration instance
config = AIAgentConfig()

# Configuration validation
def validate_and_report():
    """Validate configuration and print report"""
    issues = config.validate_config()
    
    print("🤖 AI Agent Configuration Report")
    print("=" * 50)
    
    if issues['errors']:
        print("\n❌ ERRORS:")
        for error in issues['errors']:
            print(f"   • {error}")
    
    if issues['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in issues['warnings']:
            print(f"   • {warning}")
    
    if issues['info']:
        print("\n💡 INFO:")
        for info in issues['info']:
            print(f"   • {info}")
    
    print(f"\n📊 API Status:")
    api_status = config.get_api_status()
    for api, status in api_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {api.replace('_', ' ').title()}")
    
    print("\n" + "=" * 50)
    
    return len(issues['errors']) == 0

if __name__ == "__main__":
    validate_and_report()
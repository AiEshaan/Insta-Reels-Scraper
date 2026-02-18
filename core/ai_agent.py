"""
AI Agent Core - Intelligent Instagram Content Agent
Provides autonomous reasoning, decision-making, and content analysis capabilities.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class ContentInsight:
    """Structured insights about scraped content"""
    content_id: str
    url: str
    caption: str
    sentiment: str  # positive, negative, neutral
    topics: List[str]
    engagement_prediction: float
    quality_score: float
    recommended_actions: List[str]
    timestamp: datetime

@dataclass
class AgentDecision:
    """Agent's autonomous decision with reasoning"""
    decision_type: str
    action: str
    reasoning: str
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime

class IntelligentContentAgent:
    """
    AI Agent for intelligent Instagram content management
    Features:
    - Autonomous content analysis and categorization
    - Intelligent scheduling decisions
    - Learning from user preferences
    - Trend detection and insights
    - Quality assessment and recommendations
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = None
        self._init_vector_store()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Agent tools
        self.tools = self._create_agent_tools()
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        # Learning data
        self.user_preferences = {}
        self.content_history = []
        self.decisions_history = []
        
        logger.info("AI Agent initialized successfully")
    
    def _init_vector_store(self):
        """Initialize ChromaDB vector store for content embeddings"""
        try:
            persist_directory = "output/agent_memory"
            os.makedirs(persist_directory, exist_ok=True)
            
            client = chromadb.PersistentClient(path=persist_directory)
            collection = client.get_or_create_collection("content_insights")
            
            self.vector_store = Chroma(
                client=client,
                collection_name="content_insights",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info("Vector store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None
    
    def _create_agent_tools(self) -> List[Tool]:
        """Create tools for the agent to use"""
        
        def analyze_content_sentiment(content: str) -> str:
            """Analyze sentiment of content"""
            prompt = f"""
            Analyze the sentiment of this Instagram content and classify it as positive, negative, or neutral.
            Also identify key topics and themes.
            
            Content: {content}
            
            Return a JSON with: {{"sentiment": "positive/negative/neutral", "topics": ["topic1", "topic2"], "confidence": 0.0-1.0}}
            """
            response = self.llm([HumanMessage(content=prompt)])
            return response.content
        
        def predict_engagement(content: str, topics: List[str]) -> str:
            """Predict engagement potential of content"""
            prompt = f"""
            Based on this Instagram content and topics, predict the engagement potential (0.0-1.0).
            Consider factors like trending topics, content quality, timing relevance.
            
            Content: {content}
            Topics: {topics}
            
            Return a JSON with: {{"engagement_score": 0.0-1.0, "reasoning": "explanation"}}
            """
            response = self.llm([HumanMessage(content=prompt)])
            return response.content
        
        def recommend_actions(content: str, insights: Dict) -> str:
            """Recommend actions based on content analysis"""
            prompt = f"""
            Based on this content analysis, recommend specific actions:
            
            Content: {content}
            Analysis: {insights}
            
            Suggest actions like: save_for_later, schedule_repost, analyze_further, ignore, etc.
            Return JSON with: {{"actions": ["action1", "action2"], "priority": "high/medium/low"}}
            """
            response = self.llm([HumanMessage(content=prompt)])
            return response.content
        
        return [
            Tool(
                name="analyze_sentiment",
                description="Analyze sentiment and topics of content",
                func=analyze_content_sentiment
            ),
            Tool(
                name="predict_engagement",
                description="Predict engagement potential of content",
                func=predict_engagement
            ),
            Tool(
                name="recommend_actions",
                description="Recommend actions based on content analysis",
                func=recommend_actions
            )
        ]
    
    def analyze_content(self, content_data: Dict[str, Any]) -> ContentInsight:
        """
        Perform comprehensive AI analysis of scraped content
        """
        try:
            url = content_data.get("Reel URL", "")
            caption = content_data.get("Caption", "")
            
            # Use agent to analyze content
            analysis_prompt = f"""
            Analyze this Instagram content comprehensively:
            
            URL: {url}
            Caption: {caption}
            
            Provide:
            1. Sentiment analysis (positive/negative/neutral)
            2. Key topics and themes
            3. Quality assessment (0.0-1.0)
            4. Engagement prediction (0.0-1.0)
            5. Recommended actions
            
            Be thorough and provide actionable insights.
            """
            
            analysis_result = self.agent.run(analysis_prompt)
            
            # Parse and structure the analysis
            insight = ContentInsight(
                content_id=url.split("/")[-2] if "/reel/" in url else url,
                url=url,
                caption=caption,
                sentiment="neutral",  # Default, will be updated by parsing
                topics=[],
                engagement_prediction=0.5,
                quality_score=0.5,
                recommended_actions=[],
                timestamp=datetime.now()
            )
            
            # Store in vector database for future reference
            if self.vector_store:
                self.vector_store.add_texts(
                    texts=[f"{caption} {analysis_result}"],
                    metadatas=[{"url": url, "timestamp": str(datetime.now())}]
                )
            
            self.content_history.append(insight)
            logger.info(f"Analyzed content: {url}")
            
            return insight
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return None
    
    def make_autonomous_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """
        Make autonomous decisions based on context and learned preferences
        """
        try:
            decision_prompt = f"""
            Based on the current context and situation, make an autonomous decision:
            
            Context: {json.dumps(context, indent=2)}
            User Preferences: {json.dumps(self.user_preferences, indent=2)}
            
            Consider:
            1. Content quality and relevance
            2. User's historical preferences
            3. Optimal timing and scheduling
            4. Resource allocation
            5. Strategic goals
            
            Make a decision and provide clear reasoning.
            """
            
            decision_result = self.agent.run(decision_prompt)
            
            decision = AgentDecision(
                decision_type="autonomous_action",
                action="analyze_and_categorize",  # Default action
                reasoning=decision_result,
                confidence=0.8,
                parameters=context,
                timestamp=datetime.now()
            )
            
            self.decisions_history.append(decision)
            logger.info(f"Made autonomous decision: {decision.action}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return None
    
    def learn_from_feedback(self, content_id: str, user_action: str, feedback: str = ""):
        """
        Learn from user actions and feedback to improve future decisions
        """
        try:
            learning_prompt = f"""
            Learn from this user feedback to improve future decisions:
            
            Content ID: {content_id}
            User Action: {user_action}
            Feedback: {feedback}
            
            Update preferences and decision-making patterns.
            """
            
            learning_result = self.agent.run(learning_prompt)
            
            # Update user preferences
            if user_action not in self.user_preferences:
                self.user_preferences[user_action] = []
            
            self.user_preferences[user_action].append({
                "content_id": content_id,
                "feedback": feedback,
                "timestamp": str(datetime.now())
            })
            
            logger.info(f"Learned from feedback for content: {content_id}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def get_intelligent_recommendations(self, content_batch: List[Dict]) -> List[Dict]:
        """
        Provide intelligent recommendations for a batch of content
        """
        recommendations = []
        
        for content in content_batch:
            try:
                # Analyze content
                insight = self.analyze_content(content)
                
                if insight:
                    # Make decision
                    context = {
                        "content": content,
                        "insight": asdict(insight),
                        "batch_size": len(content_batch)
                    }
                    
                    decision = self.make_autonomous_decision(context)
                    
                    recommendation = {
                        "content": content,
                        "insight": asdict(insight),
                        "decision": asdict(decision) if decision else None,
                        "priority": self._calculate_priority(insight)
                    }
                    
                    recommendations.append(recommendation)
                    
            except Exception as e:
                logger.error(f"Error processing content for recommendations: {e}")
                continue
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations
    
    def _calculate_priority(self, insight: ContentInsight) -> float:
        """Calculate priority score for content"""
        if not insight:
            return 0.0
        
        priority = (
            insight.engagement_prediction * 0.4 +
            insight.quality_score * 0.3 +
            (1.0 if insight.sentiment == "positive" else 0.5) * 0.2 +
            (len(insight.recommended_actions) / 5.0) * 0.1
        )
        
        return min(priority, 1.0)
    
    def save_agent_state(self, filepath: str = "output/agent_state.json"):
        """Save agent's learning state"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            state = {
                "user_preferences": self.user_preferences,
                "content_history": [asdict(c) for c in self.content_history[-100:]],  # Last 100
                "decisions_history": [asdict(d) for d in self.decisions_history[-100:]],  # Last 100
                "timestamp": str(datetime.now())
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Agent state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
    
    def load_agent_state(self, filepath: str = "output/agent_state.json"):
        """Load agent's learning state"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.user_preferences = state.get("user_preferences", {})
                # Note: content_history and decisions_history would need proper deserialization
                
                logger.info(f"Agent state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
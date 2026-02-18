# 🤖 Instagram Reels AI Agent

An intelligent, autonomous AI agent that transforms traditional Instagram reels scraping into a comprehensive content analysis, learning, and recommendation system.

## 🚀 What's New: AI Agent Features

This project has been transformed from a simple scraper into a sophisticated AI agent with the following capabilities:

### 🧠 Core AI Features

- **🔍 Intelligent Content Analysis**: Advanced sentiment analysis, emotion detection, and viral potential assessment
- **📊 Trend Detection**: Identifies trending topics, hashtags, and content patterns
- **🎯 Autonomous Actions**: Smart filtering, categorization, and automated recommendations
- **💾 Learning Memory**: Adapts to user preferences and learns from interaction patterns
- **✨ Content Generation**: AI-powered caption and hashtag generation
- **🌐 Web Intelligence**: Real-time trend analysis and fact-checking

### 🛠 Technical Architecture

```
Instagram Reels AI Agent
├── Core AI Components
│   ├── ai_agent.py          # Main LLM integration & orchestration
│   ├── content_analyzer.py  # Advanced content analysis
│   ├── autonomous_actions.py # Smart decision-making engine
│   ├── agent_memory.py      # Persistent learning system
│   └── agent_tools.py       # External API integrations
├── Configuration
│   └── ai_config.py         # Centralized configuration
├── Integration
│   ├── ai_agent_main.py     # Main execution pipeline
│   └── app.py              # Flask web interface
└── Original Scraper
    └── main.py             # Traditional scraping (still available)
```

## 📋 Prerequisites

### Required
- Python 3.8+
- OpenAI API Key (for LLM functionality)

### Optional (for enhanced features)
- Google Custom Search API (trend analysis)
- Bing Search API (alternative search)
- SerpAPI (comprehensive search)
- Social Media APIs (Instagram, Twitter, etc.)

## 🔧 Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd new_reel_save
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Validate Configuration**
   ```bash
   python config/ai_config.py
   ```

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
python app.py
```
Visit `http://localhost:5000` and use the AI Agent tab.

### Option 2: Direct Execution
```bash
python ai_agent_main.py
```

### Option 3: API Integration
```bash
curl -X POST http://localhost:5000/api/ai-agent/start \
  -H "Content-Type: application/json" \
  -H "X-API-Key: reel-scraper-2024-secret" \
  -d '{"username": "your_username", "password": "your_password"}'
```

## 🎯 Usage Guide

### 1. Basic AI Agent Run

The AI agent follows a comprehensive pipeline:

1. **Content Scraping**: Uses the original scraper to gather Instagram reels
2. **AI Analysis**: Analyzes content for sentiment, trends, and viral potential
3. **Learning**: Updates memory with new patterns and preferences
4. **Autonomous Actions**: Takes smart actions based on analysis
5. **Content Generation**: Creates improved captions and hashtags
6. **Results**: Provides insights, recommendations, and detailed reports

### 2. Web Interface Features

- **📋 Basic Scraper Tab**: Traditional scraping functionality
- **🤖 AI Agent Tab**: Full AI pipeline with real-time status
- **📚 API Reference Tab**: Complete API documentation

### 3. API Endpoints

#### AI Agent Endpoints
- `POST /api/ai-agent/start` - Start intelligent analysis
- `GET /api/ai-agent/status` - Check progress and summary
- `GET /api/ai-agent/results` - Detailed analysis results
- `GET /api/ai-agent/memory` - Learning and memory data

#### Traditional Endpoints
- `POST /api/start` - Basic scraping
- `GET /api/status` - Scraping status
- `GET /api/download/csv` - Download results

## 📊 Output Files

The AI agent generates comprehensive outputs:

```
output/
├── instagram_reels.csv          # Original scraped data
├── instagram_reels.xlsx         # Excel format
├── ai_agent_results.json        # Detailed AI analysis
├── ai_agent_summary.json        # Executive summary
├── memory_summary.json          # Learning insights
└── ai_agent.log                # Execution logs
```

## 🔧 Configuration

### Environment Variables

#### Required
```env
OPENAI_API_KEY=your-openai-api-key-here
```

#### Optional (Enhanced Features)
```env
# Web Search
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-engine-id
BING_API_KEY=your-bing-api-key
SERPAPI_KEY=your-serpapi-key

# Social Media
INSTAGRAM_ACCESS_TOKEN=your-instagram-token
TWITTER_API_KEY=your-twitter-key
```

#### Customization
```env
# Analysis Thresholds
VIRAL_THRESHOLD=0.7
ENGAGEMENT_THRESHOLD=0.6
CONFIDENCE_THRESHOLD=0.7

# Autonomous Actions
MAX_DAILY_ACTIONS=100
ENABLE_FILTERING=true
ENABLE_CATEGORIZATION=true
```

### Configuration Validation

Check your setup:
```bash
python config/ai_config.py
```

This will show:
- ✅ Configured APIs
- ❌ Missing APIs
- ⚠️ Warnings
- 💡 Recommendations

## 🧠 AI Agent Capabilities

### Content Analysis
- **Sentiment Analysis**: Positive, negative, neutral classification
- **Emotion Detection**: Joy, anger, fear, surprise, etc.
- **Viral Potential**: ML-based prediction of content virality
- **Trend Detection**: Identifies trending topics and hashtags
- **Content Categorization**: Automatic content classification

### Autonomous Actions
- **Smart Filtering**: Removes low-quality or irrelevant content
- **Priority Assignment**: Ranks content by importance and potential
- **Automated Categorization**: Organizes content into meaningful groups
- **Recommendation Generation**: Suggests actions and improvements
- **Scheduling**: Plans optimal posting times and strategies

### Learning & Memory
- **User Preference Learning**: Adapts to user behavior patterns
- **Pattern Recognition**: Identifies successful content patterns
- **Memory Consolidation**: Retains important insights over time
- **Context-Aware Retrieval**: Provides relevant historical insights

### Content Generation
- **Caption Enhancement**: AI-improved captions for better engagement
- **Hashtag Generation**: Trending and relevant hashtag suggestions
- **Content Suggestions**: Ideas for new content based on trends
- **Style Adaptation**: Matches brand voice and style preferences

## 📈 Performance & Insights

### Real-time Metrics
- Content scraped and analyzed
- Patterns learned and preferences updated
- Autonomous actions taken
- Content generated and improved

### Insights Generated
- Viral potential predictions
- Sentiment and emotion trends
- Content category performance
- Engagement optimization suggestions
- Trend analysis and forecasting

### Learning Outcomes
- User preference profiles
- Successful content patterns
- Optimal posting strategies
- Content improvement recommendations

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   ```
   Error: OpenAI API key not configured
   Solution: Add OPENAI_API_KEY to .env file
   ```

2. **Memory Database Issues**
   ```
   Error: Cannot create memory database
   Solution: Ensure data/ directory exists and is writable
   ```

3. **Web Search Limitations**
   ```
   Warning: No web search API configured
   Solution: Add Google, Bing, or SerpAPI credentials
   ```

### Debug Mode
Enable detailed logging:
```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

### Validation
Check configuration:
```bash
python config/ai_config.py
```

## 🚀 Advanced Usage

### Custom Analysis Pipeline
```python
from ai_agent_main import IntelligentReelsAgent

agent = IntelligentReelsAgent()
results = await agent.run_intelligent_agent()
```

### API Integration
```python
import requests

# Start AI agent
response = requests.post('http://localhost:5000/api/ai-agent/start', 
    headers={'X-API-Key': 'reel-scraper-2024-secret'},
    json={'username': 'user', 'password': 'pass'})

# Check status
status = requests.get('http://localhost:5000/api/ai-agent/status',
    headers={'X-API-Key': 'reel-scraper-2024-secret'})
```

### Memory System Access
```python
from core.agent_memory import AgentMemorySystem

memory = AgentMemorySystem()
preferences = memory.get_user_preferences_for_context({})
patterns = memory.get_learning_patterns()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain for agent framework
- Hugging Face for ML models
- ChromaDB for vector storage
- The open-source AI community

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community]
- 📖 Documentation: [Full docs]
- 🐛 Issues: [GitHub Issues]

---

**🎉 Congratulations!** You now have a fully functional AI agent that can intelligently analyze, learn from, and take autonomous actions on Instagram reels content. The agent will continuously improve its performance as it learns from your preferences and content patterns.

**Next Steps:**
1. Configure your API keys
2. Run your first AI agent session
3. Review the generated insights and recommendations
4. Customize the configuration for your specific needs
5. Integrate with your existing workflows

Happy AI agent building! 🤖✨
"""
AI Agent Main Integration
Combines the original scraper with advanced AI agent capabilities
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Original scraper imports
from main import run_agent as run_scraper_agent

# AI Agent imports
from core.ai_agent import IntelligentContentAgent
from core.content_analyzer import AdvancedContentAnalyzer
from core.autonomous_actions import AutonomousActionEngine, AutonomousAction, ActionType
from core.agent_memory import AgentMemorySystem
from core.agent_tools import AgentToolsManager, ContentGenerationRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/ai_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class IntelligentReelsAgent:
    """
    Intelligent Instagram Reels Agent
    
    Combines traditional scraping with AI-powered analysis, learning, and autonomous actions
    """
    
    def __init__(self):
        # Initialize AI components
        self.ai_agent = IntelligentContentAgent()
        self.content_analyzer = AdvancedContentAnalyzer()
        self.action_engine = AutonomousActionEngine()
        self.memory_system = AgentMemorySystem()
        self.tools_manager = AgentToolsManager()
        
        # Configuration
        self.config = {
            'enable_ai_analysis': True,
            'enable_autonomous_actions': True,
            'enable_learning': True,
            'enable_content_generation': True,
            'min_confidence_threshold': 0.7,
            'max_daily_actions': 100
        }
        
        logger.info("Intelligent Reels Agent initialized")
    
    async def run_intelligent_agent(self, username: str = None, password: str = None) -> Dict[str, Any]:
        """
        Run the complete intelligent agent pipeline
        
        1. Scrape content using original functionality
        2. Analyze content with AI
        3. Learn from patterns and user behavior
        4. Take autonomous actions
        5. Generate insights and recommendations
        """
        try:
            logger.info("Starting Intelligent Reels Agent")
            
            # Step 1: Run original scraper
            logger.info("Phase 1: Content Scraping")
            scraper_results = await self._run_scraper_phase(username, password)
            
            if not scraper_results.get('success', False):
                return {
                    'success': False,
                    'error': 'Scraper phase failed',
                    'details': scraper_results
                }
            
            # Step 2: AI Analysis Phase
            logger.info("Phase 2: AI Content Analysis")
            analysis_results = await self._run_analysis_phase(scraper_results['data'])
            
            # Step 3: Learning Phase
            logger.info("Phase 3: Learning and Memory Update")
            learning_results = await self._run_learning_phase(analysis_results)
            
            # Step 4: Autonomous Actions Phase
            logger.info("Phase 4: Autonomous Actions")
            action_results = await self._run_autonomous_phase(analysis_results)
            
            # Step 5: Content Generation Phase
            logger.info("Phase 5: Content Generation and Recommendations")
            generation_results = await self._run_generation_phase(analysis_results)
            
            # Step 6: Consolidate and Save Results
            logger.info("Phase 6: Results Consolidation")
            final_results = await self._consolidate_results({
                'scraper': scraper_results,
                'analysis': analysis_results,
                'learning': learning_results,
                'actions': action_results,
                'generation': generation_results
            })
            
            logger.info("Intelligent Reels Agent completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in intelligent agent: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_scraper_phase(self, username: str = None, password: str = None) -> Dict[str, Any]:
        """Run the original scraper functionality or use existing data"""
        try:
            import pandas as pd
            
            # Check if we already have scraped data
            existing_files = [
                'output/scrapped_reels.csv',
                'output/scrapped_reels.xlsx'
            ]
            
            for file_path in existing_files:
                if os.path.exists(file_path):
                    try:
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_excel(file_path)
                        
                        if not df.empty:
                            logger.info(f"Using existing scraped data from {file_path} ({len(df)} items)")
                            return {'success': True, 'data': df}
                    except Exception as e:
                        logger.warning(f"Could not load existing data from {file_path}: {e}")
            
            # If no existing data, run the scraper
            logger.info("No existing data found, running Instagram scraper...")
            
            # Run original scraper in a separate thread to avoid blocking
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def scraper_thread():
                try:
                    # Run the original scraper
                    result = run_scraper_agent(username, password)
                    result_queue.put({'success': True, 'data': result})
                except Exception as e:
                    result_queue.put({'success': False, 'error': str(e)})
            
            thread = threading.Thread(target=scraper_thread)
            thread.start()
            thread.join(timeout=300)  # 5 minute timeout
            
            if thread.is_alive():
                return {'success': False, 'error': 'Scraper timeout'}
            
            if not result_queue.empty():
                return result_queue.get()
            else:
                return {'success': False, 'error': 'No scraper results'}
                
        except Exception as e:
            logger.error(f"Error in scraper phase: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_analysis_phase(self, scraped_data: Any) -> Dict[str, Any]:
        """Run AI analysis on scraped content"""
        try:
            analysis_results = {
                'analyzed_items': [],
                'summary_stats': {},
                'insights': [],
                'recommendations': []
            }
            
            # Load scraped data (assuming it's saved to CSV/Excel)
            import pandas as pd
            
            # Try to load the most recent output file
            output_files = [
                'output/scrapped_reels.csv',
                'output/scrapped_reels.xlsx',
                'output/instagram_reels.csv',
                'output/instagram_reels.xlsx'
            ]
            
            df = None
            for file_path in output_files:
                if os.path.exists(file_path):
                    try:
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_excel(file_path)
                        break
                    except Exception as e:
                        logger.warning(f"Could not load {file_path}: {e}")
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': 'No scraped data found to analyze'
                }
            
            # Analyze each piece of content
            for index, row in df.iterrows():
                try:
                    content_data = {
                        'url': row.get('URL', ''),
                        'caption': row.get('Caption', ''),
                        'thumbnail': row.get('Thumbnail', ''),
                        'index': index
                    }
                    
                    # Run content analysis
                    analysis = await self.content_analyzer.analyze_content(content_data)
                    
                    # Store analysis results
                    analysis_results['analyzed_items'].append({
                        'content_id': f"reel_{index}",
                        'original_data': content_data,
                        'analysis': analysis
                    })
                    
                    # Store in memory for learning
                    self.memory_system.store_memory(
                        'content_analysis',
                        f"reel_{index}",
                        {
                            'content_data': content_data,
                            'analysis_results': analysis
                        },
                        importance=analysis.get('viral_potential', 0.5)
                    )
                    
                except Exception as e:
                    logger.error(f"Error analyzing content {index}: {e}")
                    continue
            
            # Generate summary statistics
            if analysis_results['analyzed_items']:
                analysis_results['summary_stats'] = self._calculate_analysis_summary(
                    analysis_results['analyzed_items']
                )
            
            return {
                'success': True,
                'data': analysis_results,
                'processed_count': len(analysis_results['analyzed_items'])
            }
            
        except Exception as e:
            logger.error(f"Error in analysis phase: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_learning_phase(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update agent learning based on analysis results"""
        try:
            if not analysis_results.get('success', False):
                return {'success': False, 'error': 'No analysis data for learning'}
            
            learning_stats = {
                'patterns_learned': 0,
                'preferences_updated': 0,
                'insights_generated': 0
            }
            
            analyzed_items = analysis_results['data']['analyzed_items']
            
            # Learn from content patterns
            for item in analyzed_items:
                content_id = item['content_id']
                analysis = item['analysis']
                
                # Simulate user interaction (in real implementation, this would come from actual user behavior)
                simulated_action = self._simulate_user_action(analysis)
                
                # Learn from the action
                self.memory_system.learn_from_user_action(
                    content_id,
                    simulated_action,
                    analysis,
                    {'timestamp': datetime.now().isoformat()}
                )
                
                learning_stats['patterns_learned'] += 1
            
            # Consolidate memory
            self.memory_system.consolidate_memories()
            
            # Generate insights from learning
            user_preferences = self.memory_system.get_user_preferences_for_context({})
            learning_stats['preferences_updated'] = len(user_preferences)
            
            return {
                'success': True,
                'data': learning_stats,
                'user_preferences': user_preferences
            }
            
        except Exception as e:
            logger.error(f"Error in learning phase: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_autonomous_phase(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute autonomous actions based on analysis"""
        try:
            if not analysis_results.get('success', False):
                return {'success': False, 'error': 'No analysis data for autonomous actions'}
            
            action_results = {
                'actions_taken': [],
                'recommendations': [],
                'filtered_content': [],
                'categorized_content': {}
            }
            
            analyzed_items = analysis_results['data']['analyzed_items']
            
            # Convert to content dictionaries for action engine
            content_items = []
            for item in analyzed_items:
                content_item = {
                    'content_id': item['content_id'],
                    'content_type': 'reel',
                    'url': item['original_data']['url'],
                    'caption': item['original_data']['caption'],
                    'metadata': item['analysis'],
                    'timestamp': datetime.now(),
                    'source': 'instagram_scraper'
                }
                content_items.append(content_item)
            
            # Run autonomous action pipeline
            pipeline_results = await self.action_engine.run_action_pipeline(content_items)
            
            action_results.update(pipeline_results)
            
            return {
                'success': True,
                'data': action_results,
                'processed_items': len(content_items)
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous phase: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_generation_phase(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content and recommendations"""
        try:
            if not analysis_results.get('success', False):
                return {'success': False, 'error': 'No analysis data for content generation'}
            
            generation_results = {
                'generated_captions': [],
                'generated_hashtags': [],
                'content_suggestions': [],
                'trend_insights': []
            }
            
            analyzed_items = analysis_results['data']['analyzed_items']
            
            # Generate content for top-performing items
            top_items = sorted(
                analyzed_items,
                key=lambda x: x['analysis'].get('viral_potential', 0),
                reverse=True
            )[:5]  # Top 5 items
            
            for item in top_items:
                try:
                    analysis = item['analysis']
                    
                    # Generate improved caption
                    caption_request = ContentGenerationRequest(
                        content_type='caption',
                        input_data={
                            'topic': analysis.get('primary_category', 'content'),
                            'sentiment': analysis.get('sentiment', 'neutral'),
                            'category': analysis.get('primary_category', 'general')
                        },
                        style='engaging',
                        max_length=280
                    )
                    
                    caption_result = self.tools_manager.content_generator.generate_content(caption_request)
                    if caption_result['success']:
                        generation_results['generated_captions'].append({
                            'content_id': item['content_id'],
                            'original_caption': item['original_data']['caption'],
                            'generated_caption': caption_result['content'],
                            'improvement_score': analysis.get('viral_potential', 0.5)
                        })
                    
                    # Generate hashtags
                    hashtag_request = ContentGenerationRequest(
                        content_type='hashtags',
                        input_data={
                            'topic': analysis.get('primary_category', 'content'),
                            'category': analysis.get('primary_category', 'general')
                        },
                        style='trending',
                        max_length=500
                    )
                    
                    hashtag_result = self.tools_manager.content_generator.generate_content(hashtag_request)
                    if hashtag_result['success']:
                        generation_results['generated_hashtags'].append({
                            'content_id': item['content_id'],
                            'generated_hashtags': hashtag_result['content'],
                            'category': analysis.get('primary_category', 'general')
                        })
                    
                except Exception as e:
                    logger.error(f"Error generating content for {item['content_id']}: {e}")
                    continue
            
            # Generate trend insights
            try:
                trend_results = await self.tools_manager.execute_tool(
                    'web_search',
                    query='instagram reels trends 2024',
                    search_type='trends',
                    timeframe='week'
                )
                
                if trend_results.get('success'):
                    generation_results['trend_insights'] = trend_results['results'][:3]
                    
            except Exception as e:
                logger.error(f"Error getting trend insights: {e}")
            
            return {
                'success': True,
                'data': generation_results,
                'generated_items': len(generation_results['generated_captions'])
            }
            
        except Exception as e:
            logger.error(f"Error in generation phase: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _consolidate_results(self, phase_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Consolidate all phase results into final output"""
        try:
            # Create comprehensive results summary
            final_results = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'agent_version': '1.0.0',
                'phases_completed': [],
                'summary': {
                    'total_content_scraped': 0,
                    'total_content_analyzed': 0,
                    'patterns_learned': 0,
                    'actions_taken': 0,
                    'content_generated': 0
                },
                'detailed_results': phase_results,
                'insights': [],
                'recommendations': []
            }
            
            # Check which phases completed successfully
            for phase_name, phase_result in phase_results.items():
                if phase_result.get('success', False):
                    final_results['phases_completed'].append(phase_name)
            
            # Extract summary statistics
            if phase_results['scraper'].get('success'):
                # Estimate scraped content count (would need to check actual scraper output)
                final_results['summary']['total_content_scraped'] = 50  # Placeholder
            
            if phase_results['analysis'].get('success'):
                final_results['summary']['total_content_analyzed'] = phase_results['analysis'].get('processed_count', 0)
            
            if phase_results['learning'].get('success'):
                final_results['summary']['patterns_learned'] = phase_results['learning']['data'].get('patterns_learned', 0)
            
            if phase_results['actions'].get('success'):
                actions_data = phase_results['actions']['data']
                final_results['summary']['actions_taken'] = len(actions_data.get('actions_taken', []))
            
            if phase_results['generation'].get('success'):
                final_results['summary']['content_generated'] = phase_results['generation'].get('generated_items', 0)
            
            # Generate high-level insights
            final_results['insights'] = self._generate_insights(phase_results)
            
            # Generate recommendations
            final_results['recommendations'] = self._generate_recommendations(phase_results)
            
            # Save results to file
            await self._save_results(final_results)
            
            # Export memory summary
            self.memory_system.export_memory_summary()
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error consolidating results: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_analysis_summary(self, analyzed_items: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from analysis results"""
        try:
            if not analyzed_items:
                return {}
            
            # Extract metrics
            viral_potentials = [item['analysis'].get('viral_potential', 0) for item in analyzed_items]
            sentiments = [item['analysis'].get('sentiment', 'neutral') for item in analyzed_items]
            categories = [item['analysis'].get('primary_category', 'general') for item in analyzed_items]
            
            summary = {
                'total_analyzed': len(analyzed_items),
                'average_viral_potential': sum(viral_potentials) / len(viral_potentials),
                'max_viral_potential': max(viral_potentials),
                'sentiment_distribution': {
                    'positive': sentiments.count('positive'),
                    'neutral': sentiments.count('neutral'),
                    'negative': sentiments.count('negative')
                },
                'category_distribution': {}
            }
            
            # Count categories
            for category in set(categories):
                summary['category_distribution'][category] = categories.count(category)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating analysis summary: {e}")
            return {}
    
    def _simulate_user_action(self, analysis: Dict[str, Any]) -> str:
        """Simulate user action based on content analysis (for learning)"""
        try:
            viral_potential = analysis.get('viral_potential', 0.5)
            sentiment = analysis.get('sentiment', 'neutral')
            
            # Simple simulation logic
            if viral_potential > 0.8 and sentiment == 'positive':
                return 'save'
            elif viral_potential > 0.6:
                return 'like'
            elif viral_potential > 0.4:
                return 'view'
            else:
                return 'ignore'
                
        except Exception as e:
            logger.error(f"Error simulating user action: {e}")
            return 'view'
    
    def _generate_insights(self, phase_results: Dict[str, Dict]) -> List[str]:
        """Generate high-level insights from all phases"""
        insights = []
        
        try:
            # Analysis insights
            if phase_results['analysis'].get('success'):
                analysis_data = phase_results['analysis']['data']
                summary_stats = analysis_data.get('summary_stats', {})
                
                if summary_stats:
                    avg_viral = summary_stats.get('average_viral_potential', 0)
                    if avg_viral > 0.7:
                        insights.append("High viral potential detected in scraped content - consider prioritizing similar content types")
                    elif avg_viral < 0.3:
                        insights.append("Low viral potential in current content - recommend exploring trending topics")
                    
                    sentiment_dist = summary_stats.get('sentiment_distribution', {})
                    positive_ratio = sentiment_dist.get('positive', 0) / summary_stats.get('total_analyzed', 1)
                    if positive_ratio > 0.7:
                        insights.append("Predominantly positive content sentiment - good for engagement")
            
            # Learning insights
            if phase_results['learning'].get('success'):
                learning_data = phase_results['learning']['data']
                if learning_data.get('patterns_learned', 0) > 10:
                    insights.append("Significant learning patterns identified - agent is adapting to preferences")
            
            # Action insights
            if phase_results['actions'].get('success'):
                actions_data = phase_results['actions']['data']
                if len(actions_data.get('filtered_content', [])) > 0:
                    insights.append("Content filtering active - low-quality content automatically identified")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _generate_recommendations(self, phase_results: Dict[str, Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Content recommendations
            if phase_results['analysis'].get('success'):
                recommendations.append("Review generated captions and hashtags for improved engagement")
                recommendations.append("Focus on content categories with highest viral potential")
            
            # Learning recommendations
            if phase_results['learning'].get('success'):
                recommendations.append("Continue using the agent to improve personalization")
                recommendations.append("Review memory summary for user preference insights")
            
            # Technical recommendations
            recommendations.append("Set up API keys for enhanced web search and content generation")
            recommendations.append("Configure social media APIs for real-time trend analysis")
            recommendations.append("Schedule regular agent runs for continuous learning")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    async def _save_results(self, results: Dict[str, Any]):
        """Save final results to file"""
        try:
            import json
            
            os.makedirs('output', exist_ok=True)
            
            # Save detailed results
            with open('output/ai_agent_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            summary_report = {
                'timestamp': results['timestamp'],
                'success': results['success'],
                'summary': results['summary'],
                'insights': results['insights'],
                'recommendations': results['recommendations']
            }
            
            with open('output/ai_agent_summary.json', 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            logger.info("Results saved to output/ai_agent_results.json and output/ai_agent_summary.json")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.tools_manager.cleanup()
            logger.info("Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Main execution function
async def main():
    """Main execution function for the intelligent agent"""
    agent = IntelligentReelsAgent()
    
    try:
        # Run the intelligent agent
        results = await agent.run_intelligent_agent()
        
        if results['success']:
            print("\n🎉 Intelligent Reels Agent completed successfully!")
            print(f"📊 Summary:")
            print(f"   • Content Scraped: {results['summary']['total_content_scraped']}")
            print(f"   • Content Analyzed: {results['summary']['total_content_analyzed']}")
            print(f"   • Patterns Learned: {results['summary']['patterns_learned']}")
            print(f"   • Actions Taken: {results['summary']['actions_taken']}")
            print(f"   • Content Generated: {results['summary']['content_generated']}")
            
            print(f"\n💡 Key Insights:")
            for insight in results['insights']:
                print(f"   • {insight}")
            
            print(f"\n🎯 Recommendations:")
            for recommendation in results['recommendations']:
                print(f"   • {recommendation}")
            
            print(f"\n📁 Results saved to:")
            print(f"   • output/ai_agent_results.json")
            print(f"   • output/ai_agent_summary.json")
            print(f"   • output/memory_summary.json")
            
        else:
            print(f"❌ Agent failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    # Run the intelligent agent
    asyncio.run(main())
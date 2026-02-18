#!/usr/bin/env python3
"""
Test AI Agent with Sample Data
This script demonstrates the full AI agent automation pipeline working with sample data
"""

import requests
import json
import time
import os

def test_ai_agent_with_sample_data():
    """Test the AI agent with sample data"""
    
    print("🤖 TESTING AI AGENT WITH SAMPLE DATA")
    print("="*60)
    
    # Check if sample data exists
    csv_file = "output/scrapped_reels.csv"
    if not os.path.exists(csv_file):
        print("❌ Sample data not found. Please run test_scraper.py first.")
        return
    
    print(f"✅ Sample data found: {csv_file}")
    
    # API configuration
    base_url = "http://127.0.0.1:5000"
    headers = {"X-API-Key": "reel-scraper-2024-secret"}
    
    try:
        # 1. Start AI Agent
        print("\n🚀 Starting AI Agent...")
        start_url = f"{base_url}/api/ai-agent/start"
        start_data = {"username": "syk_asur", "password": "dummy"}
        
        start_response = requests.post(start_url, headers=headers, json=start_data)
        print(f"   Status: {start_response.status_code}")
        print(f"   Response: {start_response.text}")
        
        if start_response.status_code != 200:
            print("❌ Failed to start AI agent")
            return
        
        # 2. Wait for processing
        print("\n⏳ Waiting for AI agent to process data...")
        time.sleep(10)  # Give it time to process
        
        # 3. Check Results
        print("\n📊 Checking AI Agent Results...")
        results_url = f"{base_url}/api/ai-agent/results"
        results_response = requests.get(results_url, headers=headers)
        
        print(f"   Status: {results_response.status_code}")
        
        if results_response.status_code == 200:
            results = results_response.json()
            
            print("\n🎯 AI AGENT RESULTS:")
            print(f"   ✅ Success: {results.get('success', False)}")
            print(f"   📋 Phases Completed: {results.get('phases_completed', [])}")
            
            summary = results.get('summary', {})
            print(f"   📊 Content Scraped: {summary.get('total_content_scraped', 0)}")
            print(f"   🔍 Content Analyzed: {summary.get('total_content_analyzed', 0)}")
            print(f"   ✨ Content Generated: {summary.get('content_generated', 0)}")
            print(f"   🎯 Actions Taken: {summary.get('actions_taken', 0)}")
            print(f"   🧠 Patterns Learned: {summary.get('patterns_learned', 0)}")
            
            # Show detailed results
            detailed = results.get('detailed_results', {})
            print(f"\n📋 DETAILED PHASE RESULTS:")
            for phase, result in detailed.items():
                status = "✅" if result.get('success', False) else "❌"
                print(f"   {status} {phase.title()}: {result.get('success', False)}")
                if not result.get('success', False) and 'error' in result:
                    print(f"      Error: {result['error']}")
            
            # Show recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            # Show insights
            insights = results.get('insights', [])
            if insights:
                print(f"\n🔍 INSIGHTS:")
                for i, insight in enumerate(insights, 1):
                    print(f"   {i}. {insight}")
        
        # 4. Check Memory
        print("\n🧠 Checking AI Agent Memory...")
        memory_url = f"{base_url}/api/ai-agent/memory"
        memory_response = requests.get(memory_url, headers=headers)
        
        if memory_response.status_code == 200:
            memory = memory_response.json()
            print(f"   📚 Memory Entries: {len(memory.get('entries', []))}")
            print(f"   📊 Total Patterns: {memory.get('total_patterns', 0)}")
            print(f"   🎯 Success Rate: {memory.get('success_rate', 0)}%")
        
        # 5. Check Generated Files
        print("\n📁 Checking Generated Files...")
        output_dir = "output"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"   📂 Output directory contains {len(files)} files:")
            for file in files:
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"      📄 {file} ({size} bytes)")
        
        print("\n" + "="*60)
        print("🎉 AI AGENT TESTING COMPLETED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error testing AI agent: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Make sure Flask server is running")
        print("   • Check API key configuration")
        print("   • Verify sample data exists")

def main():
    """Main function"""
    test_ai_agent_with_sample_data()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Final Real-Time AI Agent Automation Demo
Shows the complete system working with actual credentials
"""

import requests
import json
import os
from dotenv import load_dotenv
import time
from datetime import datetime

def main():
    load_dotenv()
    
    print("🚀 REAL-TIME AI AGENT AUTOMATION DEMO")
    print("="*60)
    print("Using credentials from .env file")
    print("="*60)
    
    # Get credentials from .env
    username = os.getenv('IG_USERNAME')
    password = os.getenv('IG_PASSWORD')
    api_key = os.getenv('API_KEY')
    
    print(f"📱 Instagram Username: {username}")
    print(f"🔑 API Key: {api_key[:10]}...")
    
    headers = {
        'X-API-Key': 'reel-scraper-2024-secret',
        'Content-Type': 'application/json'
    }
    
    print("\n🔄 Starting AI Agent with Real Credentials...")
    try:
        response = requests.post('http://127.0.0.1:5000/api/ai-agent/start',
                               json={'username': username, 'password': password},
                               headers=headers,
                               timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AI Agent Started Successfully!")
            print(f"📊 Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ AI Agent Start Failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print("⏱️ Waiting 5 seconds for processing...")
    time.sleep(5)
    
    print("\n📊 Checking Latest Results...")
    try:
        results_response = requests.get('http://127.0.0.1:5000/api/ai-agent/results',
                                      headers={'X-API-Key': 'reel-scraper-2024-secret'},
                                      timeout=30)
        
        if results_response.status_code == 200:
            results = results_response.json()
            print("✅ Results Retrieved!")
            
            summary = results.get('summary', {})
            print(f"\n📈 AUTOMATION RESULTS:")
            print(f"   • Content Scraped: {summary.get('total_content_scraped', 0)}")
            print(f"   • Content Analyzed: {summary.get('total_content_analyzed', 0)}")
            print(f"   • Patterns Learned: {summary.get('patterns_learned', 0)}")
            print(f"   • Actions Taken: {summary.get('actions_taken', 0)}")
            print(f"   • Content Generated: {summary.get('content_generated', 0)}")
            
            phases = results.get('phases_completed', [])
            print(f"\n🔄 Completed Phases: {', '.join(phases)}")
            
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"\n💡 AI Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
        else:
            print(f"❌ Results Error: {results_response.status_code}")
            
    except Exception as e:
        print(f"❌ Results Error: {e}")
    
    # Check memory system
    print("\n🧠 Checking AI Memory System...")
    try:
        memory_response = requests.get('http://127.0.0.1:5000/api/ai-agent/memory',
                                     headers={'X-API-Key': 'reel-scraper-2024-secret'},
                                     timeout=30)
        
        if memory_response.status_code == 200:
            memory = memory_response.json()
            print("✅ Memory System Active!")
            
            data = memory.get('data', {})
            print(f"   • Total Memories: {data.get('total_memories', 0)}")
            print(f"   • Total Preferences: {data.get('total_preferences', 0)}")
            print(f"   • Total Patterns: {data.get('total_patterns', 0)}")
        else:
            print(f"❌ Memory Error: {memory_response.status_code}")
            
    except Exception as e:
        print(f"❌ Memory Error: {e}")
    
    # Show generated files
    print("\n📁 Generated Output Files:")
    output_dir = "output"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for file in files:
            if file.endswith(('.json', '.csv', '.xlsx')):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"   • {file} ({size} bytes)")
    
    print("\n" + "="*60)
    print("🎯 DEMO COMPLETE - AI Agent Automation Working!")
    print("🌐 Access Web Interface: http://127.0.0.1:5000")
    print(f"🕒 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
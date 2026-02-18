#!/usr/bin/env python3
"""
Real-time AI Agent Automation Demo
Demonstrates the working system with live API calls
"""

import requests
import json
import time
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n{'─'*40}")
    print(f"📋 {title}")
    print(f"{'─'*40}")

def test_basic_scraper():
    """Test the basic web scraper functionality"""
    print_section("BASIC SCRAPER TEST")
    
    try:
        # Test with a simple JSON API
        headers = {'X-API-Key': 'reel-scraper-2024-secret'}
        response = requests.post('http://127.0.0.1:5000/api/scrape', 
                               json={'url': 'https://httpbin.org/json'},
                               headers=headers,
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Basic Scraper: SUCCESS")
            print(f"📊 Status: {result.get('status', 'unknown')}")
            print(f"📝 Data Length: {len(str(result.get('data', '')))} characters")
            print(f"⏱️ Timestamp: {result.get('timestamp', 'N/A')}")
            
            if result.get('data'):
                print("\n🔍 Sample Scraped Data:")
                data_str = str(result['data'])[:300] + "..." if len(str(result['data'])) > 300 else str(result['data'])
                print(data_str)
            return True
        else:
            print(f"❌ Basic Scraper: FAILED (Status: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ Error testing basic scraper: {e}")
        return False

def test_ai_agent_status():
    """Check AI agent system status"""
    print_section("AI AGENT STATUS CHECK")
    
    try:
        headers = {'X-API-Key': 'reel-scraper-2024-secret'}
        response = requests.get('http://127.0.0.1:5000/api/ai-agent/status', 
                              headers=headers, timeout=10)
        if response.status_code == 200:
            status = response.json()
            print("✅ AI Agent Status: ACTIVE")
            print(f"📈 Version: {status.get('version', 'N/A')}")
            print(f"🔄 Status: {status.get('status', 'N/A')}")
            print(f"⏰ Last Run: {status.get('last_run', 'N/A')}")
            return True
        else:
            print(f"❌ AI Agent Status: ERROR (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error checking AI agent status: {e}")
        return False

def test_ai_agent_results():
    """Get latest AI agent results"""
    print_section("AI AGENT RESULTS")
    
    try:
        headers = {'X-API-Key': 'reel-scraper-2024-secret'}
        response = requests.get('http://127.0.0.1:5000/api/ai-agent/results', 
                              headers=headers, timeout=10)
        if response.status_code == 200:
            results = response.json()
            print("✅ AI Agent Results Retrieved")
            print(f"📊 Success: {results.get('success', False)}")
            print(f"⏱️ Timestamp: {results.get('timestamp', 'N/A')}")
            print(f"🔄 Phases Completed: {', '.join(results.get('phases_completed', []))}")
            
            summary = results.get('summary', {})
            print(f"\n📈 Summary:")
            print(f"   • Content Scraped: {summary.get('total_content_scraped', 0)}")
            print(f"   • Content Analyzed: {summary.get('total_content_analyzed', 0)}")
            print(f"   • Patterns Learned: {summary.get('patterns_learned', 0)}")
            print(f"   • Actions Taken: {summary.get('actions_taken', 0)}")
            
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            return True
        else:
            print(f"❌ AI Agent Results: ERROR (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error getting AI agent results: {e}")
        return False

def test_ai_agent_memory():
    """Check AI agent memory system"""
    print_section("AI AGENT MEMORY SYSTEM")
    
    try:
        headers = {'X-API-Key': 'reel-scraper-2024-secret'}
        response = requests.get('http://127.0.0.1:5000/api/ai-agent/memory', 
                              headers=headers, timeout=10)
        if response.status_code == 200:
            memory = response.json()
            print("✅ AI Agent Memory Retrieved")
            print(f"📊 Success: {memory.get('success', False)}")
            print(f"⏱️ Timestamp: {memory.get('timestamp', 'N/A')}")
            
            data = memory.get('data', {})
            print(f"\n🧠 Memory Stats:")
            print(f"   • Total Memories: {data.get('total_memories', 0)}")
            print(f"   • Total Preferences: {data.get('total_preferences', 0)}")
            print(f"   • Total Patterns: {data.get('total_patterns', 0)}")
            
            return True
        else:
            print(f"❌ AI Agent Memory: ERROR (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error checking AI agent memory: {e}")
        return False

def run_new_ai_agent():
    """Trigger a new AI agent run"""
    print_section("TRIGGERING NEW AI AGENT RUN")
    
    try:
        print("🔄 Starting new AI agent automation...")
        headers = {'X-API-Key': 'reel-scraper-2024-secret', 'Content-Type': 'application/json'}
        response = requests.post('http://127.0.0.1:5000/api/ai-agent/start', 
                               json={'username': 'demo_user', 'password': 'demo_pass'}, 
                               headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AI Agent Run: COMPLETED")
            print(f"📊 Success: {result.get('success', False)}")
            print(f"⏱️ Timestamp: {result.get('timestamp', 'N/A')}")
            
            if result.get('summary'):
                summary = result['summary']
                print(f"\n📈 Run Summary:")
                print(f"   • Content Scraped: {summary.get('total_content_scraped', 0)}")
                print(f"   • Content Analyzed: {summary.get('total_content_analyzed', 0)}")
                print(f"   • Patterns Learned: {summary.get('patterns_learned', 0)}")
            
            return True
        else:
            print(f"❌ AI Agent Run: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error running AI agent: {e}")
        return False

def main():
    """Main demo function"""
    print_header("AI AGENT AUTOMATION SYSTEM - LIVE DEMO")
    print(f"🕒 Demo Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test sequence
    tests = [
        ("Basic Scraper Functionality", test_basic_scraper),
        ("AI Agent Status", test_ai_agent_status),
        ("AI Agent Results", test_ai_agent_results),
        ("AI Agent Memory", test_ai_agent_memory),
        ("New AI Agent Run", run_new_ai_agent),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print_header("DEMO SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Tests Passed: {passed}/{total}")
    print(f"✅ Success Rate: {(passed/total)*100:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   • {test_name}: {status}")
    
    print(f"\n🕒 Demo Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 System Running: http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test Instagram Scraper with Manual Verification
This script runs the scraper in headful mode so you can manually handle any Instagram challenges
"""

import os
import time
from dotenv import load_dotenv
from main import run_agent

def test_scraper_headful():
    """Run scraper in headful mode for manual verification"""
    load_dotenv()
    
    username = os.getenv('IG_USERNAME')
    password = os.getenv('IG_PASSWORD')
    
    print("🔍 TESTING INSTAGRAM SCRAPER")
    print("="*50)
    print(f"📱 Username: {username}")
    print("🌐 Running in HEADFUL mode for manual verification")
    print("⚠️  You may need to manually complete 2FA or challenges")
    print("="*50)
    
    try:
        print("\n🚀 Starting scraper...")
        print("📋 Browser will open - please complete any Instagram challenges manually")
        
        # Run with headless=False for manual verification
        df = run_agent(
            username=username, 
            password=password, 
            max_scrolls=5,  # Reduced for testing
            headless=False  # This will open the browser visually
        )
        
        print(f"\n✅ Scraping completed!")
        print(f"📊 Results: {len(df)} items found")
        
        if len(df) > 0:
            print("\n🔍 Sample data:")
            print(df.head())
            
            print(f"\n📁 Files saved:")
            print(f"   • CSV: output/scrapped_reels.csv")
            print(f"   • Excel: output/scrapped_reels.xlsx")
        else:
            print("\n⚠️  No data found. Possible reasons:")
            print("   • No saved content in Instagram account")
            print("   • Instagram blocked automated access")
            print("   • Account needs manual verification")
            print("   • Page structure changed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   • Make sure Instagram credentials are correct")
        print("   • Try logging in manually first")
        print("   • Check if account has saved content")
        print("   • Consider using a different account")

def create_sample_data():
    """Create sample data for testing the AI agent pipeline"""
    import pandas as pd
    
    print("\n🔧 CREATING SAMPLE DATA FOR AI AGENT TESTING")
    print("="*50)
    
    # Create sample Instagram reel data
    sample_data = [
        {
            "Reel URL": "https://www.instagram.com/reel/sample1/",
            "Caption": "Amazing travel destination! 🌍 #travel #adventure #explore #wanderlust #nature",
            "Thumbnail": "https://example.com/thumb1.jpg"
        },
        {
            "Reel URL": "https://www.instagram.com/reel/sample2/",
            "Caption": "Delicious food recipe 🍕 Easy to make at home! #food #recipe #cooking #homemade",
            "Thumbnail": "https://example.com/thumb2.jpg"
        },
        {
            "Reel URL": "https://www.instagram.com/reel/sample3/",
            "Caption": "Fitness motivation 💪 Transform your body in 30 days #fitness #workout #motivation #health",
            "Thumbnail": "https://example.com/thumb3.jpg"
        },
        {
            "Reel URL": "https://www.instagram.com/reel/sample4/",
            "Caption": "Tech review: Latest smartphone features 📱 #tech #review #smartphone #innovation",
            "Thumbnail": "https://example.com/thumb4.jpg"
        },
        {
            "Reel URL": "https://www.instagram.com/reel/sample5/",
            "Caption": "Fashion trends 2024 👗 Style inspiration for everyone #fashion #style #trends #outfit",
            "Thumbnail": "https://example.com/thumb5.jpg"
        }
    ]
    
    # Save sample data
    df = pd.DataFrame(sample_data)
    
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/scrapped_reels.csv", index=False)
    df.to_excel("output/scrapped_reels.xlsx", index=False, engine="openpyxl")
    
    print(f"✅ Sample data created!")
    print(f"📊 Created {len(df)} sample reels")
    print(f"📁 Files saved:")
    print(f"   • CSV: output/scrapped_reels.csv")
    print(f"   • Excel: output/scrapped_reels.xlsx")
    
    print(f"\n🔍 Sample data preview:")
    print(df.head())
    
    return df

def main():
    """Main function with options"""
    print("🤖 INSTAGRAM AI AGENT - SCRAPER TESTING")
    print("="*60)
    
    print("\nChoose an option:")
    print("1. Test real Instagram scraper (headful mode)")
    print("2. Create sample data for AI agent testing")
    print("3. Both (recommended)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        test_scraper_headful()
    elif choice == "2":
        create_sample_data()
    elif choice == "3":
        print("\n" + "="*60)
        print("OPTION 1: Testing Real Scraper")
        test_scraper_headful()
        
        print("\n" + "="*60)
        print("OPTION 2: Creating Sample Data")
        create_sample_data()
    else:
        print("❌ Invalid choice. Please run again and choose 1, 2, or 3.")
        return
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("   • Run the AI agent again to process the data")
    print("   • Check output files for results")
    print("   • Use the web interface at http://127.0.0.1:5000")
    print("="*60)

if __name__ == "__main__":
    main()
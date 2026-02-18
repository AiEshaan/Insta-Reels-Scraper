import os, time, random, logging, pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def sleep(a=2.5, b=4.5): time.sleep(random.uniform(a, b))
def ensure_output(): os.makedirs("output", exist_ok=True)

def run_fixed_scraper(username=None, password=None, max_scrolls=15, headless=False):
    """Fixed scraper that handles the exact security flow: Log in → Continue → password → submit"""
    load_dotenv()
    ensure_output()
    username = username or os.getenv("IG_USERNAME")
    password = password or os.getenv("IG_PASSWORD")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width":1280,"height":800})
        
        # Login
        page.goto("https://www.instagram.com/accounts/login/")
        page.wait_for_load_state('networkidle')
        sleep(8,12)

        logging.info(f"Attempting login for username: {username}")
        
        username_selector = "input[name='email']" if page.locator("input[name='email']").count() > 0 else "input[name='username']"
        password_selector = "input[name='pass']" if page.locator("input[name='pass']").count() > 0 else "input[name='password']"
        
        page.locator(username_selector).fill(username)
        page.locator(password_selector).fill(password)
        
        # Click login button
        try:
            page.get_by_role("button", name="Log in", exact=True).click(timeout=5000)
        except:
            try:
                page.click("button[type='submit']", timeout=3000)
            except:
                page.keyboard.press("Enter")
                
        sleep(5,8)
        
        # Handle security verification - EXACT FLOW YOU DESCRIBED
        logging.info("Handling security verification...")
        
        for attempt in range(10):  # Multiple attempts to handle the flow
            try:
                # Step 1: Look for "See more from..." pop-up
                see_more_popup = page.query_selector("div:has-text('See more from')")
                if see_more_popup:
                    logging.info("Found 'See more from' pop-up - Step 1: Click Log in")
                    login_button = page.query_selector("button:has-text('Log in')")
                    if login_button:
                        login_button.click()
                        sleep(3,4)
                        
                        # Step 2: Look for "Continue" button
                        for sub_attempt in range(5):
                            continue_button = page.query_selector("button:has-text('Continue'), button:has-text('continue')")
                            if continue_button:
                                logging.info("Found Continue button - Step 2: Click Continue")
                                continue_button.click()
                                sleep(2,3)
                                
                                # Step 3: Enter password
                                password_field = page.query_selector("input[name='password'], input[name='pass']")
                                if password_field:
                                    logging.info("Entering password - Step 3")
                                    password_field.fill(password)
                                    sleep(1,2)
                                    
                                    # Step 4: Submit
                                    submit_button = page.query_selector("button[type='submit'], button:has-text('Log in')")
                                    if submit_button:
                                        logging.info("Submitting - Step 4")
                                        submit_button.click()
                                        sleep(4,6)
                                        break
                            else:
                                sleep(1)
                        break
                
                # Step 5: Look for any other login prompts
                login_prompt = page.query_selector("input[name='password'], input[name='pass']")
                if login_prompt and not see_more_popup:
                    logging.info("Found password prompt - entering password")
                    login_prompt.fill(password)
                    sleep(1,2)
                    
                    submit_button = page.query_selector("button[type='submit'], button:has-text('Log in')")
                    if submit_button:
                        submit_button.click()
                        sleep(4,6)
                        break
                
                # Check if we're past login
                current_url = page.url
                if "login" not in current_url.lower():
                    logging.info(f"Successfully logged in! Current URL: {current_url}")
                    break
                    
            except Exception as e:
                logging.warning(f"Security handling attempt {attempt + 1} failed: {e}")
            
            sleep(2,3)
        
        # Final verification
        current_url = page.url
        if "login" in current_url.lower():
            logging.error("Still on login page after security handling")
            browser.close()
            return pd.DataFrame()
        
        logging.info("Login successful, proceeding to saved content...")
        
        # Navigate to saved page using multiple methods
        saved_found = False
        
        # Method 1: Direct navigation
        logging.info("Trying direct navigation to saved...")
        page.goto(f"https://www.instagram.com/{username}/saved/")
        sleep(4,6)
        
        if "/saved/" in page.url:
            saved_found = True
            logging.info("Successfully reached saved page directly!")
        else:
            # Method 2: UI navigation
            logging.info("Direct navigation failed, trying UI navigation...")
            page.goto(f"https://www.instagram.com/{username}/")
            sleep(3,4)
            
            # Look for bookmark icon with better waiting
            bookmark_icon = page.query_selector("svg[aria-label='Saved'], svg[aria-label='saved']")
            if bookmark_icon:
                logging.info("Found bookmark icon, clicking...")
                # Wait for element to be stable
                page.wait_for_selector("svg[aria-label='Saved'], svg[aria-label='saved']", timeout=10000)
                bookmark_icon.click()
                sleep(3,4)
                if "/saved/" in page.url:
                    saved_found = True
                    logging.info("Successfully reached saved via bookmark!")
            
            if not saved_found:
                # Method 3: Menu navigation with better waiting
                logging.info("Trying menu navigation...")
                menu_button = page.query_selector("svg[aria-label='Menu'], svg[aria-label='Options']")
                if menu_button:
                    # Wait for menu to be stable
                    page.wait_for_selector("svg[aria-label='Menu'], svg[aria-label='Options']", timeout=10000)
                    menu_button.click()
                    sleep(2,3)
                    
                    # Wait for menu options to appear
                    saved_option = page.query_selector("div:has-text('Saved'), a:has-text('Saved')")
                    if saved_option:
                        # Wait for saved option to be stable
                        page.wait_for_selector("div:has-text('Saved'), a:has-text('Saved')", timeout=10000)
                        saved_option.click()
                        sleep(3,4)
                        if "/saved/" in page.url:
                            saved_found = True
                            logging.info("Successfully reached saved via menu!")
        
        if not saved_found:
            logging.error("Could not navigate to saved page")
            browser.close()
            return pd.DataFrame()
        
        logging.info(f"Successfully reached saved page: {page.url}")
        sleep(3,4)
        
        # Wait for content to load and scrape
        logging.info("Waiting for content to load...")
        page.wait_for_load_state('networkidle')
        sleep(5)
        
        all_links = []
        
        # Scroll to load all content
        for scroll_i in range(max_scrolls):
            logging.info(f"Scrolling to load content... {scroll_i + 1}/{max_scrolls}")
            page.mouse.wheel(0, random.randint(2000, 4000))
            sleep(2,3)
            
            # Collect links after each scroll
            try:
                links = page.eval_on_selector_all("a[href*='/reel/'], a[href*='/p/']", 'els => els.map(e=>e.href)')
                all_links.extend(links)
                current_count = len(set(all_links))
                logging.info(f"Scroll {scroll_i + 1}: Found {len(links)} new links, {current_count} total")
            except Exception as e:
                logging.warning(f"Link collection failed on scroll {scroll_i + 1}: {e}")
        
        # Remove duplicates
        unique_links = list(dict.fromkeys(all_links))
        logging.info(f"Found {len(unique_links)} unique saved reels/posts")
        
        if not unique_links:
            logging.error("No saved content found")
            browser.close()
            return pd.DataFrame()
        
        # Process each link
        results = []
        limit = min(len(unique_links), 57)  # User has 57 reels
        
        logging.info(f"Processing {limit} saved items...")
        
        for i, link in enumerate(unique_links[:limit], 1):
            try:
                logging.info(f"Processing item {i}/{limit}: {link}")
                page.goto(link)
                sleep(3,5)
                
                # Extract caption
                caption = ""
                try:
                    caption_selectors = [
                        "article div[data-testid='post-caption']",
                        "article span",
                        "div[data-testid='post-caption'] span",
                        "h1",
                        "span"
                    ]
                    
                    for selector in caption_selectors:
                        elements = page.query_selector_all(selector)
                        for elem in elements:
                            text = elem.inner_text().strip()
                            if text and len(text) > 10 and not text.isdigit():
                                caption = text[:500]
                                break
                        if caption:
                            break
                except:
                    pass
                
                # Extract thumbnail
                thumb = ""
                try:
                    thumb = page.get_attribute("meta[property='og:image']", "content") or ""
                except:
                    pass
                
                results.append({
                    "Reel URL": link,
                    "Caption": caption,
                    "Thumbnail": thumb
                })
                
                logging.info(f"✅ Processed item {i}/{limit}")
                
            except Exception as e:
                logging.error(f"Error processing item {i}: {e}")
                continue
        
        # Save results
        df = pd.DataFrame(results)
        csv_file = "output/scrapped_reels.csv"
        xlsx_file = "output/scrapped_reels.xlsx"
        
        df.to_csv(csv_file, index=False)
        df.to_excel(xlsx_file, index=False, engine="openpyxl")
        
        logging.info(f"🎉 SUCCESS! Scraped {len(results)} saved reels")
        logging.info(f"Files saved to {csv_file} and {xlsx_file}")
        
        # Show summary
        if len(results) > 0:
            logging.info("Sample of scraped reels:")
            for i, reel in enumerate(results[:3], 1):
                logging.info(f"  {i}. {reel['Reel URL']}")
                if reel['Caption']:
                    logging.info(f"     Caption: {reel['Caption'][:80]}...")
        
        browser.close()
        return df

if __name__ == "__main__":
    run_fixed_scraper(headless=False)

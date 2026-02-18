import os, time, random, logging, pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

OUTPUT_DIR = "output"
CSV_FILE, XLSX_FILE = os.path.join(OUTPUT_DIR, "scrapped_reels.csv"), os.path.join(OUTPUT_DIR, "scrapped_reels.xlsx")

def sleep(a=2.5, b=4.5): time.sleep(random.uniform(a, b))
def ensure_output(): os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_agent(username=None, password=None, max_scrolls=10, headless=False):
    """Login → Navigate to saved → Scrape reel links, captions, thumbnails → Save CSV/XLSX"""
    load_dotenv(); ensure_output()
    username, password = username or os.getenv("IG_USERNAME"), password or os.getenv("IG_PASSWORD")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width":1280,"height":800})
        page.goto("https://www.instagram.com/accounts/login/")
        page.wait_for_load_state('networkidle')
        sleep(8,12)

        # login
        logging.info(f"Attempting login for username: {username}")
        
        # Selectors based on debug output
        username_selector = "input[name='email']" if page.locator("input[name='email']").count() > 0 else "input[name='username']"
        password_selector = "input[name='pass']" if page.locator("input[name='pass']").count() > 0 else "input[name='password']"
        
        username_field = page.locator(username_selector)
        password_field = page.locator(password_selector)

        username_field.wait_for(state='visible')
        username_field.fill(username)

        password_field.wait_for(state='visible')
        password_field.fill(password)
        
        # Try to click submit button
        try:
            # Prefer role-based selection for accessibility compliance
            page.get_by_role("button", name="Log in", exact=True).click(timeout=5000)
        except:
            try:
                page.click("button[type='submit']", timeout=3000)
            except:
                try:
                    page.click("input[type='submit']", timeout=3000)
                except:
                    page.keyboard.press("Enter")
                
        sleep(5,8)
        
        # handle various post-login scenarios and security pop-ups
        def handle_interruptions():
            # Check for "Continue" button (img1 scenario)
            try:
                if page.locator("text=Continue").is_visible():
                    logging.info("Found 'Continue' button. Clicking...")
                    page.click("text=Continue")
                    sleep(2,3)
            except:
                pass
            
            # Check for "Log in" modal (img2 scenario)
            try:
                if page.locator("[role='dialog']").is_visible() and page.locator("text=Log in").is_visible():
                    logging.info("Found 'Log in' modal. Clicking 'Log in'...")
                    page.locator("[role='dialog']").get_by_role("button", name="Log in").click()
                    sleep(2,3)
            except:
                pass
            
            # Check for password re-entry
            try:
                pass_input = page.locator("input[name='password']")
                if not pass_input.count():
                    pass_input = page.locator("input[name='pass']")
                
                if pass_input.is_visible():
                    logging.info("Password re-entry required. Filling password...")
                    pass_input.fill(password)
                    sleep(1,2)
                    
                    # Submit after password re-entry
                    submit_button = page.locator("button[type='submit'], button:has-text('Log in')")
                    if submit_button.is_visible():
                        submit_button.click()
                        sleep(3,4)
                    elif page.locator("text=Log in").is_visible():
                        page.click("text=Log in")
                        sleep(3,4)
                    else:
                        page.keyboard.press("Enter")
                        sleep(3,4)
            except:
                pass
            
            # Standard "Not Now" checks
            try:
                not_now_buttons = page.query_selector_all("text=Not Now")
                for btn in not_now_buttons:
                    btn.click()
                    sleep(1,2)
                    
                # Check for "Save Info" buttons and dismiss them
                save_info = page.query_selector("text=Save Info")
                if save_info:
                    page.click("text=Not Now")
                    sleep(1,2)
                    
                # Check for any other dialogs
                dialogs = page.query_selector_all("[role='dialog']")
                for dialog in dialogs:
                    not_now_in_dialog = dialog.query_selector("text=Not Now")
                    if not_now_in_dialog:
                        not_now_in_dialog.click()
                        sleep(1,2)
            except:
                pass

        for _ in range(3):
            handle_interruptions()
            sleep(2,3)
            
        # verify login success
        current_url = page.url
        logging.info(f"Current URL after login attempt: {current_url}")
        
        # check for error messages
        error_msg = page.query_selector("text=Sorry, your password was incorrect")
        if error_msg:
            logging.error("Login failed - incorrect password")
            browser.close()
            return pd.DataFrame()
        
        if "login" in current_url.lower():
            logging.error("Login failed - still on login page")
            if not headless:
                logging.info("It seems login failed or requires 2FA/CAPTCHA.")
                logging.info("Please complete the login manually in the browser window.")
                logging.info("Waiting for navigation away from login page...")
                
                # Wait up to 300 seconds for manual login
                for _ in range(60):
                    time.sleep(5)
                    if "login" not in page.url.lower():
                        logging.info("Login detected! Proceeding...")
                        break
                else:
                    logging.error("Still on login page after waiting. Exiting.")
                    browser.close()
                    return pd.DataFrame()
            else:
                browser.close()
                return pd.DataFrame()

        # Navigate to saved page through UI interaction
        logging.info("Navigating to saved page through UI...")
        
        # First go to profile
        page.goto(f"https://www.instagram.com/{username}/")
        sleep(4,6)
        
        # Look for and click the saved/bookmark icon
        saved_found = False
        
        # Try multiple approaches to find saved content
        for attempt in range(3):
            try:
                # Method 1: Look for bookmark icon in profile header
                bookmark_icon = page.query_selector("svg[aria-label='Saved'], svg[aria-label='saved']")
                if bookmark_icon:
                    logging.info("Found bookmark icon, clicking...")
                    bookmark_icon.click()
                    sleep(3,4)
                    if "/saved/" in page.url:
                        saved_found = True
                        break
                
                # Method 2: Look for menu button then saved option
                if not saved_found:
                    menu_button = page.query_selector("svg[aria-label='Menu'], svg[aria-label='Options']")
                    if menu_button:
                        logging.info("Clicking menu button...")
                        menu_button.click()
                        sleep(2,3)
                        
                        # Wait for menu to open
                        page.wait_for_selector("div:has-text('Saved'), a:has-text('Saved')", timeout=5000)
                        
                        # Look for saved option in menu
                        saved_option = page.query_selector("div:has-text('Saved'), a:has-text('Saved')")
                        if saved_option:
                            logging.info("Found saved option in menu, clicking...")
                            saved_option.click()
                            sleep(3,4)
                            if "/saved/" in page.url:
                                saved_found = True
                                break
                
                # Method 3: Direct navigation attempt
                if not saved_found:
                    logging.info("Trying direct navigation to saved...")
                    page.goto(f"https://www.instagram.com/{username}/saved/")
                    sleep(4,6)
                    if "/saved/" in page.url:
                        saved_found = True
                        break
                        
            except Exception as e:
                logging.warning(f"Navigation attempt {attempt + 1} failed: {e}")
                sleep(2,3)
        
        if not saved_found:
            logging.error("Could not navigate to saved page")
            browser.close()
            return pd.DataFrame()
            
        logging.info(f"Successfully reached saved page: {page.url}")
        sleep(3,4)
        
        # try multiple selectors for saved folders
        folder_links = []
        selectors = [
            "a[href*='/saved/all-posts/']",
            "a[href*='/saved/']",
            "a[href*='/saved/all/']",
            "div[role='button'] a[href*='/saved/']",
            "svg[aria-label='Saved'] + a",
            "a[href*='/collections/']"
        ]
        
        for selector in selectors:
            try:
                links = page.eval_on_selector_all(selector, 'els => els.map(e=>e.href)')
                if links:
                    folder_links = links
                    break
            except:
                continue
                
        folder_links = list(dict.fromkeys(folder_links))
        logging.info(f"Found {len(folder_links)} saved folders")
        
        if len(folder_links) == 0:
            logging.warning("No saved folders found.")
            # Check for logout
            if page.locator("text=Log In").count() > 0 or page.locator("text=Log in").count() > 0:
                logging.warning("Detected logout. Please log in manually in the browser.")
                for _ in range(60):
                    time.sleep(5)
                    if page.locator("text=Log In").count() == 0 and page.locator("text=Log in").count() == 0:
                        logging.info("Login detected! Refreshing...")
                        page.goto(f"https://www.instagram.com/{username}/saved/"); sleep(4,6)
                        # Re-run selectors
                        for selector in selectors:
                            try:
                                links = page.eval_on_selector_all(selector, 'els => els.map(e=>e.href)')
                                if links:
                                    folder_links = links
                                    break
                            except:
                                continue
                        break
            
            if not folder_links:
                logging.warning("Still no saved folders. Dumping page for debugging.")
                try:
                    page.screenshot(path="saved_page_debug.png")
                    with open("saved_page_debug.html", "w", encoding="utf-8") as f:
                        f.write(page.content())
                except Exception as e:
                    logging.error(f"Failed to save debug info: {e}")

        links = []
        # scrape each folder
        for folder in folder_links:
            page.goto(folder); sleep(4,6)
            
            # Enhanced content loading strategy
            def wait_for_content():
                """Wait for dynamic content to load with multiple strategies"""
                # Wait for network idle
                page.wait_for_load_state('networkidle')
                sleep(2)
                
                # Wait for specific content indicators
                for wait_attempt in range(3):
                    try:
                        # Look for any post/reel elements
                        content_elements = page.query_selector_all("article, a[href*='/reel/'], a[href*='/p/']")
                        if content_elements:
                            logging.info(f"Content loaded! Found {len(content_elements)} content elements")
                            return True
                        
                        # Look for grid layout indicators
                        grid_indicators = page.query_selector_all("div[class*='grid'], div[class*='vsc']")
                        if grid_indicators:
                            logging.info(f"Grid layout detected! Found {len(grid_indicators)} grid elements")
                            return True
                        
                        # Scroll to trigger content loading
                        page.mouse.wheel(0, 1000)
                        sleep(2)
                        
                    except:
                        pass
                    
                return False
            
            # Enhanced link collection with better filtering
            def collect_links_enhanced():
                """Collect links with better filtering and retry logic"""
                all_links = []
                
                # Multiple scroll attempts to ensure content loads
                for scroll_attempt in range(5):
                    page.mouse.wheel(0, random.randint(2000, 4000))
                    sleep(2)
                    
                    # Collect links after each scroll
                    try:
                        current_links = page.eval_on_selector_all("a", "els => els.map(e=>e.href)")
                        # Filter for reel/post links
                        filtered_links = [link for link in current_links if link and ('/reel/' in link or '/p/' in link)]
                        all_links.extend(filtered_links)
                        
                        # Remove duplicates
                        unique_links = list(dict.fromkeys(all_links))
                        logging.info(f"Scroll {scroll_attempt + 1}: Found {len(filtered_links)} new links, {len(unique_links)} total unique")
                        
                        # Stop if we're finding lots of links (likely all loaded)
                        if len(unique_links) > 50:  # Assuming user has around 57 reels max
                            logging.info("Content appears fully loaded!")
                            return unique_links
                            
                    except Exception as e:
                        logging.warning(f"Link collection attempt {scroll_attempt + 1} failed: {e}")
                
                return list(dict.fromkeys(all_links))
            
            # Wait for content to load
            if not wait_for_content():
                logging.warning("Failed to load content, skipping folder")
                continue
            
            # Use enhanced collection method
            reel_links = collect_links_enhanced()
            links.extend(reel_links)
            logging.info(f"Found {len(reel_links)} items in folder {folder}")
        
        links = list(dict.fromkeys(links))
        logging.info(f"Found {len(links)} total unique items")

        results=[]
        for i, link in enumerate(links, 1):
            try:
                page.goto(link); sleep(4,6)
                caption = page.inner_text("article")[:500] if page.query_selector("article") else ""
                thumb = page.get_attribute("meta[property='og:image']","content") or ""
                results.append({"Reel URL":link,"Caption":caption,"Thumbnail":thumb})
                logging.info(f"Processed reel {i}/{len(links)}")
            except Exception as e:
                logging.error(f"Error processing {link}: {str(e)}")
                continue

        # save
        df = pd.DataFrame(results)
        df.to_csv(CSV_FILE,index=False); df.to_excel(XLSX_FILE,index=False,engine="openpyxl")
        browser.close()
        return df
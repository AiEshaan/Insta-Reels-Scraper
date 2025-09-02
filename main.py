import os, time, random, logging, pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

OUTPUT_DIR = "output"
CSV_FILE, XLSX_FILE = os.path.join(OUTPUT_DIR, "scrapped_reels.csv"), os.path.join(OUTPUT_DIR, "scrapped_reels.xlsx")

def sleep(a=2.5, b=4.5): time.sleep(random.uniform(a, b))
def ensure_output(): os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_agent(username=None, password=None, max_scrolls=10, headless=True):
    """Login → Navigate to saved → Scrape reel links, captions, thumbnails → Save CSV/XLSX"""
    load_dotenv(); ensure_output()
    username, password = username or os.getenv("IG_USERNAME"), password or os.getenv("IG_PASSWORD")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width":1280,"height":800})
        page.goto("https://www.instagram.com/accounts/login/")
        sleep(4,6)

        # login
        page.fill("input[name='username']", username)
        page.fill("input[name='password']", password)
        page.click("button[type='submit']"); sleep(8,10)

        # get saved folders
        page.goto(f"https://www.instagram.com/{username}/saved/"); sleep(4,6)
        folder_links = page.eval_on_selector_all("a[href*='/saved/']",'els => els.map(e=>e.href)')
        folder_links = list(dict.fromkeys(folder_links))
        logging.info(f"Found {len(folder_links)} saved folders")
        
        links = []
        # scrape each folder
        for folder in folder_links:
            page.goto(folder); sleep(4,6)
            for _ in range(max_scrolls):
                page.mouse.wheel(0, random.randint(2000,3000))
                sleep(2,3)
                logging.info(f"Scrolling folder {folder}... ({_+1}/{max_scrolls})")
            
            # collect reel URLs from this folder
            folder_links = page.eval_on_selector_all("a", "els => els.map(e=>e.href)")
            folder_links = list(dict.fromkeys([l for l in folder_links if "/reel/" in l or "/p/" in l]))
            links.extend(folder_links)
            logging.info(f"Found {len(folder_links)} items in folder {folder}")
        
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
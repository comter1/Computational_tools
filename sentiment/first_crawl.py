#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 00:48:44 2025

@author: liuxiaosa
"""

"""
twitter_all_in_one.py — DOM extraction for Twitter/X:
Extracts username, text content, timestamp, and optional links inside tweet text.
- Fully rewritten to use XPath (to avoid unsupported CSS selectors such as :has)
- Saves daily CSV output (even if zero results) and stores debug HTML/PNG files
- Output directory is fixed to data/ under the script's directory

For academic and research purposes only. Follow all platform Terms of Service
and applicable local laws.
"""

import csv, time, datetime
from pathlib import Path
from urllib.parse import quote
import pandas as pd

from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# User-adjustable configuration
USE_TWITTER_DOMAIN   = False    # False = x.com, True = twitter.com
HEADLESS             = False    # Recommended False at first so the page is visible
KEYWORD_CSV          = "keyword.csv"   # First column name must be: (keyword)
LANG                 = "en"
SINCE                = "2020-01-22"    # Start date (inclusive)
UNTIL                = "2021-03-01"    # End date (exclusive)
EXCLUDE_RETWEETS     = True

STOP_NUM_PER_QUERY   = 500      # Max tweets per day
SCROLL_TIMES_PER_DAY = 250      # Number of scrolls per day
SCROLL_INTERVAL_SEC  = 1.3      # Scroll interval
DEBUG                = True     # Save debug HTML/PNG
WRITE_EMPTY_DAY      = True     # Write CSV even when zero results

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/125.0.0.0 Safari/537.36")

# Create and configure Chrome WebDriver instance
def create_driver(headless=False) -> webdriver.Chrome:
    opts = ChromeOptions()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--start-maximized")
    opts.add_argument(f"--user-agent={UA}")
    if headless:
        opts.add_argument("--headless=new")
    drv = webdriver.Chrome(options=opts)
    drv.set_page_load_timeout(90)
    drv.implicitly_wait(5)
    return drv

# Ensure the user is logged in before scraping
def ensure_logged_in(driver, use_twitter_domain=False):
    base = "https://twitter.com" if use_twitter_domain else "https://x.com"
    driver.get(base + "/")
    time.sleep(2)
    page = driver.page_source
    if ("Log in" in page) or ("登录" in page) or ("Sign in" in page):
        print("Please complete login in the opened browser, then return here and press Enter to continue.")
        input()

# Generate daily date ranges
def day_slices(since: str, until: str):
    d = datetime.datetime.strptime(since, "%Y-%m-%d")
    end = datetime.datetime.strptime(until, "%Y-%m-%d")
    while d < end:
        s = d.strftime("%Y-%m-%d")
        u = (d + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        yield s, u
        d += datetime.timedelta(days=1)

# Build search query string
def build_query(keyword: str, lang: str, s: str, u: str, exclude_rt=True) -> str:
    q = f"({keyword}) lang:{lang} since:{s} until:{u}"
    if exclude_rt:
        q += " -is:retweet"
    return q

# Build URL for the constructed query
def build_url(q: str, twitter_domain=False) -> str:
    base = "https://twitter.com" if twitter_domain else "https://x.com"
    return f"{base}/search?q={quote(q)}&src=typed_query&f=live"

# Normalize text: remove line breaks, tabs, extra whitespace
def norm(s: str | None) -> str:
    return (s or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

# Parse a single tweet <article> block using XPath
def parse_article(article):
    """
    Returns a dictionary:
      {
        "user": "screen_name",
        "text": "full text",
        "datetime": "ISO8601 timestamp",
        "tweet_id": "...",
        "links": ["https://...", ...]  # URLs in the tweet text (optional)
      }
    or None if parsing fails.
    """
    try:
        # Extract timestamp
        t = article.find_element(By.XPATH, './/time[@datetime]')
        created_at = t.get_attribute("datetime") or ""

        # Extract tweet_id from /status/ link
        tweet_id = ""
        id_a = article.find_elements(By.XPATH, './/a[contains(@href,"/status/")]')
        for a in id_a:
            href = (a.get_attribute("href") or "")
            if "/status/" in href:
                tweet_id = href.split("/status/")[-1].split("?")[0]
                break

        # Extract username in multiple fallback ways
        user = ""
        # 1) Preferred: infer from /<user>/status/<id> link structure
        if id_a:
            href = (id_a[0].get_attribute("href") or "").split("?")[0]
            parts = [p for p in href.split("/") if p]
            if "status" in parts:
                idx = parts.index("status")
                if idx - 1 >= 0:
                    user = parts[idx - 1]

        # 2) Fallback: the first user link under User-Name container
        if not user:
            try:
                ua = article.find_element(By.XPATH, './/div[@data-testid="User-Name"]//a[starts-with(@href,"/")]')
                href = (ua.get_attribute("href") or "").split("?")[0]
                seg = [p for p in href.split("/") if p]
                if seg:
                    user = seg[-1]
            except Exception:
                pass

        # 3) Fallback: visible @handle
        if not user:
            try:
                handle = article.find_element(By.XPATH, './/div[@data-testid="User-Name"]//span[starts-with(normalize-space(.),"@")]').text
                user = handle.lstrip("@")
            except Exception:
                pass

        # Extract tweet text and URLs in text
        text = ""
        links = []
        try:
            text_div = article.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
            text = norm(text_div.text)
            link_els = text_div.find_elements(By.XPATH, './/a[@href]')
            links = [a.get_attribute("href") for a in link_els if a.get_attribute("href")]
        except Exception:
            pass  # Some tweets may be image-only with no text container

        if user and created_at:
            return {
                "user": user,
                "text": text,
                "datetime": created_at,
                "tweet_id": tweet_id,
                "links": links
            }
        return None
    except Exception:
        return None

# Scroll page and collect tweets up to limits
def scroll_and_collect(driver, stop_num=200, scroll_times=40, scroll_interval=1.6):
    rows, seen = [], set()

    # Wait for the first tweet article to appear
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//article[.//time[@datetime]]'))
        )
    except Exception:
        pass

    # Basic diagnostic checks
    txt = driver.page_source
    if ("Log in" in txt) or ("登录" in txt) or ("Sign in" in txt):
        print("The current page appears to be a login page. Please log in first.")
    if "Something went wrong" in txt:
        print("An error occurred on the page. Scrolling will continue.")

    # Parse visible tweets
    def grab_current():
        nonlocal rows
        arts = driver.find_elements(By.XPATH, '//article[.//time[@datetime]]')
        for a in arts:
            rec = parse_article(a)
            if not rec:
                continue
            key = rec["tweet_id"] or (rec["user"], rec["datetime"], rec["text"][:40])
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
            if len(rows) >= stop_num:
                return True
        return False

    # First screen capture
    grab_current()

    # Scroll loop
    for _ in range(scroll_times):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(scroll_interval)
        if grab_current():
            break

    return rows

# Save tweet rows to CSV file
def save_rows_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["User_name", "Datetime", "Content", "Links", "Tweet_ID"])
        for r in rows:
            w.writerow([
                r["user"],
                r["datetime"],
                r["text"],
                " ".join(r["links"]) if r["links"] else "",
                r["tweet_id"]
            ])

# Read keywords from CSV
def read_keywords(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="GB18030")

# Main workflow
def run(driver):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("OUTPUT_DIR =", OUTPUT_DIR.resolve())

    df = read_keywords(KEYWORD_CSV)
    if "keyword" not in df.columns:
        raise ValueError("Column 'keyword' is missing in keyword.csv")

    for kw in df["keyword"]:
        for s, u in day_slices(SINCE, UNTIL):
            q = build_query(kw, LANG, s, u, exclude_rt=EXCLUDE_RETWEETS)
            url = build_url(q, twitter_domain=USE_TWITTER_DOMAIN)
            print(f"[{s} ~ {u})  {q}")
            driver.get(url)
            time.sleep(2)

            rows = scroll_and_collect(
                driver,
                stop_num=STOP_NUM_PER_QUERY,
                scroll_times=SCROLL_TIMES_PER_DAY,
                scroll_interval=SCROLL_INTERVAL_SEC
            )

            day_stub = f"{s}_{u}"

            # Save debug HTML and PNG for this date
            if DEBUG:
                (OUTPUT_DIR / f"debug_{day_stub}.html").write_text(driver.page_source, encoding="utf-8")
                try:
                    driver.save_screenshot(str(OUTPUT_DIR / f"debug_{day_stub}.png"))
                except Exception:
                    pass

            # Daily CSV file (all keywords for that day appended to same file)
            csv_path = OUTPUT_DIR / f"{day_stub}.csv"

            if rows:
                save_rows_csv(rows, csv_path)
                print(f"Wrote {len(rows)} records → {csv_path.name}")
            else:
                if WRITE_EMPTY_DAY:
                    if not csv_path.exists() or csv_path.stat().st_size == 0:
                        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                            w = csv.writer(f)
                            w.writerow(["User_name","Datetime","Content","Links","Tweet_ID"])
                    print(f"0 records → Empty CSV written: {csv_path.name}")
                else:
                    print("0 records (file not written)")

if __name__ == "__main__":
    driver = create_driver(headless=HEADLESS)
    ensure_logged_in(driver, use_twitter_domain=USE_TWITTER_DOMAIN)
    run(driver)
    driver.quit()
    print("Done.")

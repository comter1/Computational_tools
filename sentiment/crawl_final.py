#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 21:05:45 2025

@author: liuxiaosa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
twitter_all_in_one.py — DOM 抓取：用户名、文本、时间（可选：文本里的链接）
- 彻底改用 XPath（避免 :has 选择器不兼容）
- 每天都落 CSV（即使 0 条），并保存 debug HTML/PNG 便于定位
- 输出目录固定为脚本同目录的 data/
仅供学习研究；请遵守站点条款与当地法律。
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

# ========= 你只需改这里（必要时） =========
USE_TWITTER_DOMAIN   = False    # False=x.com，True=twitter.com
HEADLESS             = False    # 首次建议 False（可看见页面）
KEYWORD_CSV          = "keyword.csv"   # 第一列列名必须是：关键词
LANG                 = "en"
SINCE                = "2020-10-22"    # 起始（含）
UNTIL                = "2021-01-01"    # 终止（不含）
EXCLUDE_RETWEETS     = True

STOP_NUM_PER_QUERY   = 500      # 每日最多抓多少条
SCROLL_TIMES_PER_DAY = 250       # 每天下拉次数
SCROLL_INTERVAL_SEC  = 1.3      # 下拉间隔
DEBUG                = True     # 保存 debug_*.html/png
WRITE_EMPTY_DAY      = True     # 0 条也写表头
# ======================================

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/125.0.0.0 Safari/537.36")

# ----------------- 浏览器 -----------------
def create_driver(headless=False) -> webdriver.Chrome:
    opts = ChromeOptions()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--start-maximized")
    opts.add_argument(f"--user-agent={UA}")
    if headless:
        opts.add_argument("--headless=new")
    drv = webdriver.Chrome(options=opts)  # Selenium Manager 自动匹配驱动
    drv.set_page_load_timeout(90)
    drv.implicitly_wait(5)
    return drv

def ensure_logged_in(driver, use_twitter_domain=False):
    base = "https://twitter.com" if use_twitter_domain else "https://x.com"
    driver.get(base + "/")
    time.sleep(2)
    page = driver.page_source
    if ("Log in" in page) or ("登录" in page) or ("Sign in" in page):
        print("请在打开的浏览器里完成登录，然后回到终端按回车继续…")
        input()

# ----------------- 工具 -----------------
def day_slices(since: str, until: str):
    d = datetime.datetime.strptime(since, "%Y-%m-%d")
    end = datetime.datetime.strptime(until, "%Y-%m-%d")
    while d < end:
        s = d.strftime("%Y-%m-%d")
        u = (d + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        yield s, u
        d += datetime.timedelta(days=1)

def build_query(keyword: str, lang: str, s: str, u: str, exclude_rt=True) -> str:
    q = f"({keyword}) lang:{lang} since:{s} until:{u}"
    if exclude_rt:
        q += " -is:retweet"
    return q

def build_url(q: str, twitter_domain=False) -> str:
    base = "https://twitter.com" if twitter_domain else "https://x.com"
    return f"{base}/search?q={quote(q)}&src=typed_query&f=live"

def norm(s: str | None) -> str:
    return (s or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

# ----------------- 解析 1 条推文（XPath 版） -----------------
def parse_article(article):
    """
    返回:
      {
        "user": "screen_name",
        "text": "整条文本",
        "datetime": "ISO8601",
        "tweet_id": "...",
        "links": ["https://...", ...]  # 文本中的 URL（可为空）
      } 或 None
    """
    try:
        # 时间
        t = article.find_element(By.XPATH, './/time[@datetime]')
        created_at = t.get_attribute("datetime") or ""

        # tweet_id
        tweet_id = ""
        id_a = article.find_elements(By.XPATH, './/a[contains(@href,"/status/")]')
        for a in id_a:
            href = (a.get_attribute("href") or "")
            if "/status/" in href:
                tweet_id = href.split("/status/")[-1].split("?")[0]
                break

        # 用户名（handle）
        user = ""
        # 1) 优先从带 /status 的链接回溯：.../<user>/status/<id>
        if id_a:
            href = (id_a[0].get_attribute("href") or "").split("?")[0]
            parts = [p for p in href.split("/") if p]
            # [..., domain, user, status, id] 或 [https:, '', x.com, user, status, id]
            if "status" in parts:
                idx = parts.index("status")
                if idx-1 >= 0:
                    user = parts[idx-1]
        # 2) 退回到「User-Name」容器的第一个 /<user> 链接
        if not user:
            try:
                ua = article.find_element(By.XPATH, './/div[@data-testid="User-Name"]//a[starts-with(@href,"/")]')
                href = (ua.get_attribute("href") or "").split("?")[0]
                seg = [p for p in href.split("/") if p]
                if seg:
                    user = seg[-1]
            except Exception:
                pass
        # 3) 再退回到显示的 @handle 文本
        if not user:
            try:
                handle = article.find_element(By.XPATH, './/div[@data-testid="User-Name"]//span[starts-with(normalize-space(.),"@")]').text
                user = handle.lstrip("@")
            except Exception:
                pass

        # 文本 & 文本内链接
        text = ""
        links = []
        try:
            text_div = article.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
            text = norm(text_div.text)
            link_els = text_div.find_elements(By.XPATH, './/a[@href]')
            links = [a.get_attribute("href") for a in link_els if a.get_attribute("href")]
        except Exception:
            # 纯图片等没有 tweetText 时，给空文本
            pass

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

# ----------------- 滚动收集 -----------------
def scroll_and_collect(driver, stop_num=200, scroll_times=40, scroll_interval=1.6):
    rows, seen = [], set()

    # 等待首条 article（XPath 版）
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//article[.//time[@datetime]]'))
        )
    except Exception:
        pass

    # 诊断
    txt = driver.page_source
    if ("Log in" in txt) or ("登录" in txt) or ("Sign in" in txt):
        print("⚠️ 当前页是登录页，请先登录。")
    if "Something went wrong" in txt:
        print("⚠️ 页面报错，将继续尝试滚动。")

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

    # 首屏
    grab_current()

    # 滚动
    for _ in range(scroll_times):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(scroll_interval)
        if grab_current():
            break

    return rows

# ----------------- 落盘 -----------------
def save_rows_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            # 只写你要的：用户名、文本、时间、（可选）文本里的链接、tweet_id
            w.writerow(["User_name", "Datetime", "Content", "Links", "Tweet_ID"])
        for r in rows:
            w.writerow([
                r["user"],
                r["datetime"],
                r["text"],
                " ".join(r["links"]) if r["links"] else "",
                r["tweet_id"]
            ])

# ----------------- 主流程 -----------------
def read_keywords(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="GB18030")

def run(driver):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("OUTPUT_DIR =", OUTPUT_DIR.resolve())

    df = read_keywords(KEYWORD_CSV)
    if "关键词" not in df.columns:
        raise ValueError("keyword.csv 缺少列：关键词")

    for kw in df["关键词"]:
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
# 仅用日期做文件名（不含关键词）
            day_stub = f"{s}_{u}"

# 调试快照（按天命名）
            if DEBUG:
                (OUTPUT_DIR / f"debug_{day_stub}.html").write_text(driver.page_source, encoding="utf-8")
                try:
                    driver.save_screenshot(str(OUTPUT_DIR / f"debug_{day_stub}.png"))
                except Exception:
                    pass

# 当天的 CSV（同一天内所有关键词都写到这一个文件里）
            csv_path = OUTPUT_DIR / f"{day_stub}.csv"
            
            if rows:
                save_rows_csv(rows, csv_path)
                print(f" 写入 {len(rows)} 条 → {csv_path.name}")
            else:
                if WRITE_EMPTY_DAY:
                    if not csv_path.exists() or csv_path.stat().st_size == 0:
                        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                            w = csv.writer(f); w.writerow(["User_name","Datetime","Content","Links","Tweet_ID"])
                    print(f"  0 条 → 已写空 CSV：{csv_path.name}")
                else:
                    print("  0 条（未写文件）")

if __name__ == "__main__":
    driver = create_driver(headless=HEADLESS)
    ensure_logged_in(driver, use_twitter_domain=USE_TWITTER_DOMAIN)
    run(driver)
    driver.quit()
    print("Done.")

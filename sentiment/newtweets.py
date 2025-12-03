#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 22:22:58 2025

@author: liuxiaosa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_clean_text.py
读取所有原始爬虫 CSV → 文本清洗 → 输出 processed_tweets.csv
"""

import re
import pandas as pd
import glob
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# ================
# 文本清洗函数
# ================
def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    # 去 URL
    text = re.sub(r"http\S+", "", text)

    # 去 @用户名
    text = re.sub(r"@\w+", "", text)

    # 去 hashtag 的 # 符号
    text = text.replace("#", "")

    # 去 email
    text = re.sub(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", "", text)

    # 去 emoji
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+", 
        flags=re.UNICODE)
    text = emoji_pattern.sub("", text)

    # 统一小写
    text = text.lower()

    # 去标点
    text = re.sub(r"[^\w\s]", " ", text)

    # 去多空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ================
# 读取所有 CSV
# ================
def load_all_csv(pattern="data/2020-*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("❌ 没找到 data/2020-*.csv")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ================
# 主程序
# ================
if __name__ == "__main__":
    print("加载原始推文 CSV...")
    df = load_all_csv("data/2020-*.csv")

    # 确保列名正确
    if "Content" not in df.columns:
        raise ValueError("❌ 原始 CSV 缺少 Content 列（推文正文）")

    print("清洗文本...")
    df["clean_text"] = df["Content"].astype(str).apply(clean_text)

    print("保存到 processed_tweets.csv...")
    df.to_csv("processed_tweets.csv", index=False, encoding="utf-8")

    print("✔ 完成！已生成：processed_tweets.csv")

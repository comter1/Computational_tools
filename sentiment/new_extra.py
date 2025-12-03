#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_tweets_compare.py
文本清洗 + TextRank vs KeyBERT 关键词抽取 + N-gram 去重 + 可视化
"""

import re
import pandas as pd
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
import glob
import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from scipy.stats import entropy

# -------------------------
# 初始化模型
# -------------------------
kw_model = KeyBERT(model=SentenceTransformer('all-MiniLM-L6-v2'))
nltk.download('punkt')
nltk.download("wordnet")
nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# =============================================
# N-gram 归一化（核心：自动去除重复短语）
# =============================================
def normalize_ngram(phrase):
    """统一归一 n-gram，消除重复：
       - 小写
       - 分词
       - 去停用词
       - 词形还原（cases→case）
       - 按字母排序（coronavirus china == china coronavirus）
    """
    if not isinstance(phrase, str):
        return ""

    tokens = phrase.lower().split()
    tokens = [
        lemm.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in stop_words
    ]

    if not tokens:
        return ""

    tokens = sorted(tokens)  # 短语无序 → 有序
    return " ".join(tokens)


# =============================================
# 文本清洗
# =============================================
def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    text = re.sub(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", "", text)

    emoji_pattern = re.compile("[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================
# TextRank
# =============================================
def combine(word_list, window=2):
    for x in range(1, window):
        if x >= len(word_list):
            break
        for a, b in zip(word_list, word_list[x:]):
            yield a, b

def textrank(block_words, topK=5):
    G = nx.Graph()
    for word_list in block_words:
        for u, v in combine(word_list, 2):
            G.add_edge(u, v)

    if len(G.nodes()) == 0:
        return []

    pr = nx.pagerank(G)
    return sorted(pr.items(), key=lambda x: x[1], reverse=True)[:topK]

def extract_textrank_keywords(df, text_col="clean_text"):
    raw_scores = Counter()

    for text in df[text_col]:
        words = word_tokenize(text)
        valid = [w for w in words if w.isalpha() and len(w) > 2]
        if not valid:
            continue

        for w, s in textrank([valid]):
            raw_scores[w] += s

    # ---- 加入 n-gram 去重 ----
    normalized_scores = Counter()
    for phrase, score in raw_scores.items():
        key = normalize_ngram(phrase)
        if key:
            normalized_scores[key] += score

    return dict(normalized_scores)


# =============================================
# KeyBERT
# =============================================
def extract_keybert_keywords(df, text_col="clean_text"):
    raw_scores = Counter()
    MIN_LEN = 4

    for text in df[text_col]:
        if not isinstance(text, str) or not text.strip():
            continue

        try:
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=5
            )
        except Exception:
            continue

        for phrase, score in kws:
            phrase = phrase.lower().strip()
            if len(phrase) < MIN_LEN:
                continue
            raw_scores[phrase] += score

    # ---- 加入 n-gram 去重 ----
    normalized_scores = Counter()
    for phrase, score in raw_scores.items():
        key = normalize_ngram(phrase)
        if key:
            normalized_scores[key] += score

    return dict(normalized_scores)


# =============================================
# 可视化
# =============================================
def plot_wordcloud(freq_dict, title, filename):
    wc = WordCloud(width=800, height=400, background_color="white")
    wc.generate_from_frequencies(freq_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_bar(freq_dict, title, filename, top_n=20):
    items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, values = zip(*items)
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def kl_divergence(p_dict, q_dict):
    all_keys = list(set(p_dict.keys()) | set(q_dict.keys()))
    p = np.array([p_dict.get(k, 1e-9) for k in all_keys])
    q = np.array([q_dict.get(k, 1e-9) for k in all_keys])
    return entropy(p, q)


# =============================================
# 加载文件
# =============================================
def load_all_csv(pattern="data1/2020-*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("没找到 CSV 文件！")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


# =============================================
# 主程序入口
# =============================================
if __name__ == "__main__":
    print("加载 CSV 文件...")
    df = load_all_csv("data1/2020-*.csv")

    print("清洗文本...")
    df["clean_text"] = df["Content"].astype(str).apply(clean_text)

    print("提取 TextRank 关键词...")
    kw_textrank = extract_textrank_keywords(df)

    print("提取 KeyBERT 关键词...")
    kw_keybert = extract_keybert_keywords(df)

    # 保存结果
    pd.Series(kw_textrank).sort_values(ascending=False).to_csv("keywords_textrank.csv")
    pd.Series(kw_keybert).sort_values(ascending=False).to_csv("keywords_keybert.csv")

    print("生成词云...")
    plot_wordcloud(kw_textrank, "TextRank Keywords", "wordcloud_textrank.png")
    plot_wordcloud(kw_keybert, "KeyBERT Keywords", "wordcloud_keybert.png")

    print("生成柱状图...")
    plot_bar(kw_textrank, "Top TextRank Keywords", "bar_textrank.png")
    plot_bar(kw_keybert, "Top KeyBERT Keywords", "bar_keybert.png")

    print("计算 KL-Divergence...")
    kl_score = kl_divergence(kw_textrank, kw_keybert)
    with open("kl_divergence.txt", "w") as f:
        f.write(f"KL Divergence (TextRank || KeyBERT): {kl_score}\n")

    print("\n==============================")
    print("✔ 任务完成！")
    print("输出包括：")
    print("- keywords_textrank.csv")
    print("- keywords_keybert.csv")
    print("- wordcloud_textrank.png")
    print("- wordcloud_keybert.png")
    print("- bar_textrank.png")
    print("- bar_keybert.png")
    print("- kl_divergence.txt")
    print("==============================")

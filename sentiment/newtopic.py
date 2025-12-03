#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_modeling.py

基于 BERTweet 句向量的主题建模与聚类评估：
1) 比较 KMeans / Agglomerative / HDBSCAN 聚类质量（Silhouette / DBI / CH）
2) 使用 BERTopic 进行主题建模（基于预计算 embedding）
3) 保存：
   - clustering_eval.csv             各聚类方法指标
   - topic_info.csv                  主题列表与关键词
   - tweets_with_topics.csv          每条推文的主题分配
   - topic_trend_top10.png           按周的前10个主题热度曲线
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import hdbscan
from bertopic import BERTopic


# ==============================
# 1. 读取数据与 embedding
# ==============================
TEXT_COL = "clean_text"
TIME_COL = "Datetime"

TWEET_FILE = "processed_tweets.csv"
EMB_FILE   = "tweet_embeddings_BERTWEET.npy"   # 你之前生成的 embedding 文件

print("📂 Loading tweets and embeddings...")
df = pd.read_csv(TWEET_FILE)
embeddings = np.load(EMB_FILE)

if TEXT_COL not in df.columns:
    raise ValueError(f"列 {TEXT_COL} 不存在，请检查 processed_tweets.csv")

if len(df) != embeddings.shape[0]:
    raise ValueError(f"样本数不匹配：df={len(df)} vs embeddings={embeddings.shape[0]}")


# ==============================
# 2. 聚类评估：KMeans / Agglomerative / HDBSCAN
# ==============================
def evaluate_clustering_models(embeddings, max_samples=5000, random_state=42):
    """
    对同一批 embedding 使用不同聚类算法，比较：
    - Silhouette Score
    - Davies-Bouldin Score
    - Calinski-Harabasz Score
    """
    n_samples = embeddings.shape[0]
    if n_samples > max_samples:
        idx = np.random.RandomState(random_state).choice(n_samples, max_samples, replace=False)
        X = embeddings[idx]
    else:
        X = embeddings

    results = []

    def safe_metrics(X, labels, name):
        """计算聚类指标（处理 label 全相同、噪声过多的情况）"""
        labels = np.array(labels)
        # 过滤 HDBSCAN 中的噪声点
        mask = labels != -1
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            print(f"⚠ {name}: 有效聚类太少，无法计算指标。")
            return {
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
                "n_clusters": len(np.unique(labels[mask])),
                "noise_ratio": float((labels == -1).mean())
            }

        Xv = X[mask]
        lv = labels[mask]

        try:
            sil = silhouette_score(Xv, lv)
        except Exception:
            sil = np.nan
        try:
            dbi = davies_bouldin_score(Xv, lv)
        except Exception:
            dbi = np.nan
        try:
            ch = calinski_harabasz_score(Xv, lv)
        except Exception:
            ch = np.nan

        return {
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch,
            "n_clusters": len(np.unique(lv)),
            "noise_ratio": float((labels == -1).mean())
        }

    # ---------- KMeans ----------
    for k in [20, 30, 40]:
        print(f"🔹 Evaluating KMeans(k={k}) ...")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        m = safe_metrics(X, labels, f"KMeans_k={k}")
        m["model"] = "KMeans"
        m["param"] = f"k={k}"
        results.append(m)

    # ---------- Agglomerative ----------
    for k in [20, 30]:
        print(f"🔹 Evaluating Agglomerative(n_clusters={k}) ...")
        ag = AgglomerativeClustering(n_clusters=k)
        labels = ag.fit_predict(X)
        m = safe_metrics(X, labels, f"Agglomerative_k={k}")
        m["model"] = "Agglomerative"
        m["param"] = f"k={k}"
        results.append(m)

    # ---------- HDBSCAN ----------
    for min_cluster_size in [15, 30]:
        print(f"🔹 Evaluating HDBSCAN(min_cluster_size={min_cluster_size}) ...")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        labels = hdb.fit_predict(X)
        m = safe_metrics(X, labels, f"HDBSCAN_mcs={min_cluster_size}")
        m["model"] = "HDBSCAN"
        m["param"] = f"min_cluster_size={min_cluster_size}"
        results.append(m)

    eval_df = pd.DataFrame(results)
    return eval_df


print("\n📊 Evaluating clustering models (KMeans / Agglomerative / HDBSCAN)...")
eval_df = evaluate_clustering_models(embeddings)
eval_df.to_csv("clustering_eval.csv", index=False)
print("✔ Saved: clustering_eval.csv")
print(eval_df)


# ==============================
# 3. 使用 BERTopic 进行主题建模
# ==============================
print("\n🚀 Fitting BERTopic with precomputed embeddings...")

docs = df[TEXT_COL].astype(str).tolist()

# 让 BERTopic 使用我们提供的 embedding（不再自己编码）
topic_model = BERTopic(
    embedding_model=None,   # 不用内置的 embedding
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

df["topic_id"] = topics
df["topic_prob"] = probs

# 主题信息（ID、大小、关键词）
topic_info = topic_model.get_topic_info()
topic_info.to_csv("topic_info.csv", index=False, encoding="utf-8")
df.to_csv("tweets_with_topics.csv", index=False, encoding="utf-8")

print("✔ Saved: topic_info.csv")
print("✔ Saved: tweets_with_topics.csv")
print(f"共得到 {len(topic_info)} 个 topic（包含 -1 噪声）")


# ==============================
# 4. 对 BERTopic 结果做聚类指标评估
# ==============================
print("\n📐 Evaluating BERTopic topic assignments...")

labels = np.array(topics)
mask = labels != -1

if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
    Xv = embeddings[mask]
    lv = labels[mask]

    try:
        sil_bertopic = silhouette_score(Xv, lv)
    except Exception:
        sil_bertopic = np.nan

    try:
        dbi_bertopic = davies_bouldin_score(Xv, lv)
    except Exception:
        dbi_bertopic = np.nan

    try:
        ch_bertopic = calinski_harabasz_score(Xv, lv)
    except Exception:
        ch_bertopic = np.nan
else:
    sil_bertopic = dbi_bertopic = ch_bertopic = np.nan

# 尝试计算主题一致性（coherence）
try:
    coherence = topic_model.get_coherence()
except Exception:
    coherence = np.nan

with open("bertopic_eval.txt", "w", encoding="utf-8") as f:
    f.write(f"Silhouette (BERTopic topics): {sil_bertopic}\n")
    f.write(f"Davies-Bouldin (BERTopic topics): {dbi_bertopic}\n")
    f.write(f"Calinski-Harabasz (BERTopic topics): {ch_bertopic}\n")
    f.write(f"Topic Coherence (BERTopic): {coherence}\n")

print("✔ Saved: bertopic_eval.txt")
print("BERTopic clustering quality:")
print(f"  Silhouette       = {sil_bertopic}")
print(f"  Davies-Bouldin   = {dbi_bertopic}")
print(f"  Calinski-Harabasz= {ch_bertopic}")
print(f"  Coherence        = {coherence}")


# ==============================
# 5. 主题时间趋势（按周发帖量）
# ==============================
print("\n📈 Computing topic trends over time...")

# 解析时间
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL])

df["week"] = df[TIME_COL].dt.to_period("W").dt.start_time

# 去掉噪声 topic -1
df_valid = df[df["topic_id"] != -1].copy()

topic_counts = (
    df_valid
    .groupby(["week", "topic_id"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

# 选取整体上最热的前 10 个 topic
total_counts = topic_counts.sum(axis=0).sort_values(ascending=False)
top_topics = total_counts.head(10).index.tolist()

topic_counts_top = topic_counts[top_topics]

plt.figure(figsize=(12, 6))
for t in top_topics:
    plt.plot(topic_counts_top.index, topic_counts_top[t], label=f"Topic {t}")

plt.xlabel("Week")
plt.ylabel("Tweet Count")
plt.title("Top 10 Topics - Weekly Volume")
plt.legend(loc="upper right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("topic_trend_top10.png", dpi=300)
plt.close()

print("✔ Saved: topic_trend_top10.png")
print("\n🎉 All topic modeling steps completed!")

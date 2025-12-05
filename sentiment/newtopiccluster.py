#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST TOPIC MODELING PIPELINE
UMAP → 10% Sampling → BERTopic
No full transform. Analyze sample only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import umap
import hdbscan
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



TEXT_COL = "clean_text"
TIME_COL = "Datetime"
TWEET_FILE = "processed_tweets.csv"
EMB_FILE  = "tweet_embeddings_MINILM.npy"   # Embeddings must match rows in csv

print(" Loading tweets & embeddings...")
df_raw = pd.read_csv(TWEET_FILE)
emb_raw = np.load(EMB_FILE)

df_raw[TIME_COL] = pd.to_datetime(df_raw[TIME_COL], errors="coerce")

# Filter period
mask = (df_raw[TIME_COL] >= "2020-01-01") & (df_raw[TIME_COL] <= "2021-02-28")
df  = df_raw[mask].copy()
embeddings = emb_raw[mask.values]

print(f" Remaining tweets after filtering: {len(df)}")



print("\n Checking UMAP embedding file...")

if os.path.exists("emb_red.npy"):
    print("⚡ Using cached emb_red.npy")
    emb_red = np.load("emb_red.npy")
else:
    print(" Running UMAP for first time... may take ~10-20min on M1")
    umap_model = umap.UMAP(
        n_neighbors=30,
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    emb_red = umap_model.fit_transform(embeddings)
    np.save("emb_red.npy", emb_red)
    print("Saved UMAP result emb_red.npy")

print("UMAP output shape:", emb_red.shape)



SAMPLE_RATIO = 0.10   # Fast & recommended

print(f"\n Sampling {SAMPLE_RATIO*100:.0f}% of dataset...")

if os.path.exists("sample_idx.npy"):
    print("⚡ Loading saved sample_idx.npy")
    idx = np.load("sample_idx.npy")
else:
    idx = np.random.choice(len(emb_red), int(len(emb_red)*SAMPLE_RATIO), replace=False)
    np.save("sample_idx.npy", idx)
    print(" Saved indices → sample_idx.npy")

emb_sample  = emb_red[idx]
docs_sample = df[TEXT_COL].iloc[idx].astype(str).fillna("").tolist()   # 🔥 FIX: avoid float→str error

print(f" Sample size: {len(idx)}")



print("\n Training BERTopic on 10% sample...")

vectorizer_model = CountVectorizer(stop_words="english", min_df=20)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

topic_model = BERTopic(
    embedding_model=None,   # We already use reduced embeddings
    umap_model=None,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    low_memory=True,
    calculate_probabilities=True,
    verbose=True
)

topics_sample, probs_sample = topic_model.fit_transform(
    tqdm(docs_sample, desc="Fitting topic model", ncols=90),
    embeddings=emb_sample
)

print(" BERTopic training completed ⚡")


df_sample = df.iloc[idx].copy()
df_sample["topic_id"]   = topics_sample
df_sample["topic_prob"] = probs_sample.max(axis=1)

df_sample.to_csv("tweets_topics_SAMPLE.csv", index=False)
topic_model.get_topic_info().to_csv("topic_info_SAMPLE.csv", index=False)

print(" Results saved:")
print("   tweets_topics_SAMPLE.csv")
print("   topic_info_SAMPLE.csv")


print("\n Calculating clustering metrics...")

mask_valid = np.array(topics_sample) != -1
Xv = emb_sample[mask_valid]
lv = np.array(topics_sample)[mask_valid]

try: sil = silhouette_score(Xv, lv)
except: sil = np.nan
try: dbi = davies_bouldin_score(Xv, lv)
except: dbi = np.nan
try: ch  = calinski_harabasz_score(Xv, lv)
except: ch = np.nan

with open("bertopic_sample_eval.txt", "w") as f:
    f.write(f"Silhouette Score: {sil}\n")
    f.write(f"Davies-Bouldin Score: {dbi}\n")
    f.write(f"Calinski-Harabasz Score: {ch}\n")

print(" Evaluation saved → bertopic_sample_eval.txt")



print("\n Generating Topic Trend Plot...")

df_sample["week"] = df_sample[TIME_COL].dt.to_period("W").dt.start_time
df_valid = df_sample[df_sample["topic_id"] != -1]

topic_counts = df_valid.groupby(["week","topic_id"]).size().unstack(fill_value=0)
top_topics = topic_counts.sum().sort_values(ascending=False).head(10).index

plt.figure(figsize=(12,6))
for t in top_topics:
    plt.plot(topic_counts.index, topic_counts[t], label=f"Topic {t}")

plt.title("Top 10 Topics - Weekly Trend (Sample Only)")
plt.xlabel("Week")
plt.ylabel("Tweet Count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("topic_trend_sample_top10.png", dpi=300)
plt.close()

print("Figure saved → topic_trend_sample_top10.png")
print("\n DONE — Fast Topic Modeling Completed Successfully!\n")

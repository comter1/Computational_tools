#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embedding_generator.py
生成 tweet 的句向量（Sentence-BERT & BERTweet）
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. 读取清洗后的推文
# ============================================================
INPUT_FILE = "processed_tweets.csv"    # 你之前生成的文件
df = pd.read_csv(INPUT_FILE)

if "clean_text" not in df.columns:
    raise ValueError("❌ ERROR: processed_tweets.csv 中没有 'clean_text' 列，请先运行清洗脚本。")

texts = df["clean_text"].astype(str).tolist()
print(f"📌 Loaded {len(texts)} tweets for embedding.")


# ============================================================
# 2. 定义生成 embedding 的函数
# ============================================================
def generate_embeddings(model_name, texts, batch_size=64):
    print(f"\n🚀 Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = []
    print(f"⚙ Generating embeddings using {model_name} ...")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    print(f"✔ Shape = {embeddings.shape}")
    return embeddings


# ============================================================
# 3. 生成两个模型的句向量
# ============================================================

# ---- Sentence-BERT：MiniLM (fast) ----
emb_minilm = generate_embeddings("all-MiniLM-L6-v2", texts)
np.save("tweet_embeddings_MINILM.npy", emb_minilm)
print("💾 Saved: tweet_embeddings_MINILM.npy")

# ---- BERTweet：Tweet-specific model ----
emb_bertweet = generate_embeddings("vinai/bertweet-base", texts)
np.save("tweet_embeddings_BERTWEET.npy", emb_bertweet)
print("💾 Saved: tweet_embeddings_BERTWEET.npy")


# ============================================================
# 4. 保存 CSV（只保存向量的前 10 维，便于查看）
# ============================================================
df_out = df.copy()

# 添加前10维到 CSV
for i in range(10):
    df_out[f"minilm_dim_{i}"] = emb_minilm[:, i]
    df_out[f"bertweet_dim_{i}"] = emb_bertweet[:, i]

df_out.to_csv("tweets_with_embeddings.csv", index=False, encoding="utf-8")
print("💾 Saved: tweets_with_embeddings.csv (first 10 dims only)")

print("\n🎉 All embeddings generated successfully!")

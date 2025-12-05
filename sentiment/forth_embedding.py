#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 01:01:40 2025

@author: liuxiaosa
"""

"""
Generate sentence embeddings for tweets (Sentence-BERT & BERTweet)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load cleaned tweet dataset
INPUT_FILE = "processed_tweets.csv"
df = pd.read_csv(INPUT_FILE)

# Ensure required column exists
if "clean_text" not in df.columns:
    raise ValueError("processed_tweets.csv missing 'clean_text'. Please run text cleaning first.")

texts = df["clean_text"].astype(str).tolist()
print(f"Loaded {len(texts)} tweets for embedding generation.")

# Generate embeddings using a selected model
def generate_embeddings(model_name, texts, batch_size=64):
    """
    Generate sentence embeddings in batch mode to avoid memory overflow.
    You can change the model_name to use other models such as:
        - vinai/bertweet-base (tweet-specific)
        - all-MiniLM-L6-v2 (fast sentence embedding model)
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = []
    print(f"Generating embeddings using {model_name} ...")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


# Generate embeddings using two models

# BERTweet model (pretrained for tweets/social media text)
emb_bertweet = generate_embeddings("vinai/bertweet-base", texts)
np.save("tweet_embeddings_BERTWEET.npy", emb_bertweet)
print("Saved: tweet_embeddings_BERTWEET.npy")

# Sentence-BERT MiniLM model (fast general embedding model)
emb_minilm = generate_embeddings("all-MiniLM-L6-v2", texts)
np.save("tweet_embeddings_MINILM.npy", emb_minilm)
print("Saved: tweet_embeddings_MINILM.npy")


# Preview embeddings by exporting first 10 dimensions into CSV
df_out = df.copy()

# Append the first 10 dimensions for each embedding model
for i in range(10):
    df_out[f"minilm_dim_{i}"] = emb_minilm[:, i]
    df_out[f"bertweet_dim_{i}"] = emb_bertweet[:, i]

df_out.to_csv("tweets_with_embeddings.csv", index=False, encoding="utf-8")
print("Saved: tweets_with_embeddings.csv (first 10 dimensions included)")

print("\nAll embeddings generated successfully.")

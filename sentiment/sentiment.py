#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment_analysis.py
生成推文情感分析：
1) RoBERTa Sentiment（正/中/负）
2) GoEmotions（27类情绪 + 中性）
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ==============================
# 1. 加载清洗后的推文
# ==============================
INPUT_FILE = "processed_tweets.csv"

df = pd.read_csv(INPUT_FILE)
texts = df["clean_text"].astype(str).tolist()

print(f"📌 Loaded {len(texts)} tweets for sentiment analysis.")


# ==============================
# 2. RoBERTa Twitter Sentiment
# ==============================
print("\n🚀 Loading RoBERTa Twitter Sentiment model...")

sent_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)

sent_model.eval()

LABELS_ROBERTA = ["negative", "neutral", "positive"]

def sentiment_roberta(texts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        encoded = sent_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = sent_model(**encoded).logits

        probs = F.softmax(logits, dim=-1).cpu().numpy()

        for p in probs:
            idx = np.argmax(p)
            results.append({
                "sentiment_label": LABELS_ROBERTA[idx],
                "sent_neg": p[0],
                "sent_neu": p[1],
                "sent_pos": p[2],
                "sent_max": p[idx],
            })
    return results


print("\n⚙ Running RoBERTa sentiment...")
sent_results = sentiment_roberta(texts)

sent_df = pd.DataFrame(sent_results)
print("✔ RoBERTa sentiment completed.")


# ==============================
# 3. GoEmotions 情绪分类
# ==============================
print("\n🚀 Loading GoEmotions model...")

go_model_name = "bhadresh-savani/bert-base-go-emotions"
go_tokenizer = AutoTokenizer.from_pretrained(go_model_name)
go_model = AutoModelForSequenceClassification.from_pretrained(go_model_name)

go_model.eval()

# 官方的 28 类情绪标签
GOEMO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

def goemotions_predict(texts, batch_size=16):
    emo_list = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        encoded = go_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = go_model(**encoded).logits

        probs = F.softmax(logits, dim=-1).cpu().numpy()

        for p in probs:
            idx = np.argmax(p)
            emo_record = {
                "main_emotion": GOEMO_LABELS[idx],
                "main_emotion_score": p[idx]
            }
            # 每个维度的概率都保存
            for j, emo_name in enumerate(GOEMO_LABELS):
                emo_record[f"emo_{emo_name}"] = p[j]
            emo_list.append(emo_record)

    return emo_list


print("\n⚙ Running GoEmotions...")
go_results = goemotions_predict(texts)
go_df = pd.DataFrame(go_results)
print("✔ GoEmotions completed.")


# ==============================
# 4. 合并并保存
# ==============================
print("\n💾 Saving results...")

df_out = pd.concat([df, sent_df, go_df], axis=1)
df_out.to_csv("tweets_with_sentiment.csv", index=False, encoding="utf-8")

print("🎉 DONE: tweets_with_sentiment.csv")
print("包含：RoBERTa 情感 + GoEmotions 27 类情绪")

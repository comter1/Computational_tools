#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment_analysis.py
Generate sentiment and emotion labels for tweets:
1) RoBERTa sentiment classification (positive / neutral / negative)
2) GoEmotions emotion classification (27 emotions + neutral)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load cleaned text dataset
INPUT_FILE = "processed_tweets.csv"

df = pd.read_csv(INPUT_FILE)
texts = df["clean_text"].astype(str).tolist()

print(f"Loaded {len(texts)} tweets for sentiment analysis.")


print("\nLoading RoBERTa Twitter sentiment model...")

# Local pretrained RoBERTa sentiment model
sent_model_name = "./local_models/roberta_sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)

sent_model.eval()

# Class labels for RoBERTa model output
LABELS_ROBERTA = ["negative", "neutral", "positive"]

def sentiment_roberta(texts, batch_size=32):
    """
    Run RoBERTa sentiment classification on a list of texts.
    Returns a list of dictionaries with:
        - sentiment_label (negative / neutral / positive)
        - sent_neg, sent_neu, sent_pos (probabilities)
        - sent_max (max probability among the three classes)
    """
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


print("\nRunning RoBERTa sentiment analysis...")
sent_results = sentiment_roberta(texts)

sent_df = pd.DataFrame(sent_results)
print("RoBERTa sentiment analysis completed.")


print("\nLoading GoEmotions model...")

# Local pretrained GoEmotions model
go_model_name = "./local_models/go_emotions"
go_tokenizer = AutoTokenizer.from_pretrained(go_model_name)
go_model = AutoModelForSequenceClassification.from_pretrained(go_model_name)

go_model.eval()

# Official GoEmotions label set (27 emotions + neutral)
GOEMO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

def goemotions_predict(texts, batch_size=16):
    """
    Run GoEmotions classification on a list of texts.
    For each text, returns:
        - main_emotion: label with highest probability
        - main_emotion_score: probability of the main_emotion
        - emo_<emotion_name>: probability for each of the 28 emotion labels
    """
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
            # Save the probability for each emotion dimension
            for j, emo_name in enumerate(GOEMO_LABELS):
                emo_record[f"emo_{emo_name}"] = p[j]
            emo_list.append(emo_record)

    return emo_list


print("\nRunning GoEmotions classification...")
go_results = goemotions_predict(texts)
go_df = pd.DataFrame(go_results)
print("GoEmotions classification completed.")


print("\nSaving results...")

# Merge original data with sentiment and emotion outputs
df_out = pd.concat([df, sent_df, go_df], axis=1)
df_out.to_csv("tweets_with_sentiment.csv", index=False, encoding="utf-8")

print("Done: tweets_with_sentiment.csv saved.")
print("Contents: RoBERTa sentiment scores and GoEmotions emotion distribution.")

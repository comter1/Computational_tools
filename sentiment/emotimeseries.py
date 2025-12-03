#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emotion_timeseries.py
基于 tweets_with_sentiment.csv
绘制：
1) RoBERTa 正/负/中 情绪折线图
2) GoEmotions 28维情绪热力图
3) 主导情绪时间序列
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1. 读取情感分析数据
# =======================
df = pd.read_csv("tweets_with_sentiment.csv")

# 时间格式
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

# 按周聚合
df["week"] = df["Datetime"].dt.to_period("W").dt.start_time

print(f"Loaded {len(df)} tweets with sentiment.")


# =======================
# 2. RoBERTa 情绪趋势
# =======================
roberta_cols = ["sent_neg", "sent_neu", "sent_pos"]

weekly_roberta = df.groupby("week")[roberta_cols].mean()

plt.figure(figsize=(12, 5))
plt.plot(weekly_roberta.index, weekly_roberta["sent_pos"], label="Positive")
plt.plot(weekly_roberta.index, weekly_roberta["sent_neu"], label="Neutral")
plt.plot(weekly_roberta.index, weekly_roberta["sent_neg"], label="Negative")

plt.title("RoBERTa Weekly Sentiment Trend")
plt.xlabel("Week")
plt.ylabel("Average Sentiment Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_weekly_roberta_sentiment.png", dpi=300)
plt.close()

print("✔ Saved: plot_weekly_roberta_sentiment.png")


# =======================
# 3. GoEmotions 28 类热力图
# =======================
emotion_cols = [c for c in df.columns if c.startswith("emo_")]

weekly_emotions = df.groupby("week")[emotion_cols].mean()

plt.figure(figsize=(14, 10))
sns.heatmap(
    weekly_emotions.T,
    cmap="coolwarm",
    linewidths=0.3
)

plt.title("GoEmotions - Weekly Emotion Heatmap")
plt.xlabel("Week")
plt.ylabel("Emotion")
plt.tight_layout()
plt.savefig("plot_goemotions_heatmap.png", dpi=300)
plt.close()

print("✔ Saved: plot_goemotions_heatmap.png")


# =======================
# 4. 主导情绪时间序列
# =======================
# main_emotion 来自 GoEmotions
weekly_major = df.groupby("week")["main_emotion"].agg(lambda x: x.value_counts().index[0])

plt.figure(figsize=(12, 5))
plt.plot(
    weekly_major.index,
    weekly_major.values,
    marker="o",
    linestyle="-"
)
plt.title("Weekly Major Emotion (GoEmotions)")
plt.xlabel("Week")
plt.ylabel("Dominant Emotion")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_major_emotion_timeline.png", dpi=300)
plt.close()

print("✔ Saved: plot_major_emotion_timeline.png")
print("🎉 All emotion time-series plots generated!")

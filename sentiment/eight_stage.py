#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage_analysis.py

Stage-based comparison for COVID-related tweet dynamics

Functions included:
1) Load sentiment + topic assignments and ensure ID consistency
2) Merge datasets using tweet_id (strict column normalization)
3) Load weekly COVID data and perform automatic stage segmentation (ruptures)
4) Map tweets to pandemic stages
5) Compute topic frequency per stage
6) Perform emotion significance tests across stages
7) Run lagged cross-correlation between topic frequency & COVID case numbers
8) Stage visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
from scipy.stats import ttest_ind, ks_2samp
from statsmodels.tsa.stattools import ccf


print("Loading topic and sentiment data...")

df_sent = pd.read_csv("tweets_with_sentiment.csv", low_memory=False)
df_topics = pd.read_csv("tweets_with_topics.csv", low_memory=False)

# Normalize tweet_id column names to ensure merge compatibility
rename_map = {"Tweet_ID": "tweet_id", "TweetId": "tweet_id", "id": "tweet_id"}
df_sent.rename(columns=rename_map, inplace=True)
df_topics.rename(columns=rename_map, inplace=True)

# Convert ID to str to prevent mismatched merge cases
df_sent["tweet_id"] = df_sent["tweet_id"].astype(str).strip()
df_topics["tweet_id"] = df_topics["tweet_id"].astype(str).strip()

# Remove duplicate tweet entries if exist
df_sent = df_sent.drop_duplicates(subset="tweet_id")
df_topics = df_topics.drop_duplicates(subset="tweet_id")

# Merge sentiment with topic assignment
df = df_sent.merge(
    df_topics[["tweet_id", "topic_id", "topic_prob"]],
    on="tweet_id",
    how="left"
)

# Convert topic field to numeric; drop noise topics (-1)
df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce")
df = df[df["topic_id"].notna()]
df = df[df["topic_id"] != -1]

print("Final merged tweets:", len(df))
print(df.head())

# Convert datetime field
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.dropna(subset=["Datetime"])
df["week"] = df["Datetime"].dt.to_period("W").dt.start_time

print("Datetime parsed and weekly index generated.")

print("Loading COVID weekly dataset...")
covid = pd.read_csv("clean_covid_weekly.csv")
covid["week"] = pd.to_datetime(covid["week"], errors="coerce")
print("COVID data loaded.")

# Stage segmentation using ruptures breakpoint detection
print("Running ruptures change-point detection...")

series = covid["daily_new_cases"].values
algo = rpt.Pelt(model="rbf").fit(series)
raw_breaks = algo.predict(pen=5)

# Filter invalid breakpoints
raw_breaks = sorted([b for b in raw_breaks if 0 < b < len(covid)])
boundaries = [0] + raw_breaks + [len(covid)]

covid["stage"] = 0
for i in range(len(boundaries)-1):
    covid.loc[boundaries[i]:boundaries[i+1], "stage"] = i + 1

print("Stage segmentation completed.")

# Map tweets to stage label
df = df.merge(
    covid[["week", "stage", "daily_new_cases"]],
    on="week",
    how="left"
)

print("Tweets matched to stage labels.")

# Stage-level topic frequency
print("Computing stage-based topic frequency...")

topic_freq = df.groupby(["stage", "topic_id"]).size().reset_index(name="count")
topic_freq.to_csv("output_stage_topic_frequency.csv", index=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=topic_freq, x="stage", y="count", hue="topic_id")
plt.title("Topic Frequency per Stage")
plt.savefig("plot_stage_topic_freq.png", dpi=300)
plt.close()

# Emotion significance across stages (t-test + KS test)
print("Performing emotion significance test...")

emotions = ["sent_pos", "sent_neu", "sent_neg"]
results = []
stages = sorted(df["stage"].dropna().unique())

for e in emotions:
    for i in range(len(stages)-1):
        s1 = df[df["stage"] == stages[i]][e].dropna()
        s2 = df[df["stage"] == stages[i+1]][e].dropna()
        results.append([
            e,
            stages[i],
            stages[i+1],
            ttest_ind(s1, s2, equal_var=False).pvalue,
            ks_2samp(s1, s2).pvalue
        ])

pd.DataFrame(
    results,
    columns=["emotion", "stage1", "stage2", "t_test_p", "ks_test_p"]
).to_csv("output_emotion_significance.csv", index=False)

print("Emotion significance test completed.")

# Cross-correlation for lag analysis
print("Computing lagged cross-correlation...")

weekly_topic = df.groupby(["week", "topic_id"]).size().unstack(fill_value=0)
aligned = covid.set_index("week")[["daily_new_cases"]].join(weekly_topic, how="left").fillna(0)

corr_results = []
lags = range(-4, 5)

for topic in weekly_topic.columns:
    t_series = aligned[topic].values
    c_series = aligned["daily_new_cases"].values
    cc = ccf(t_series, c_series)
    for lag in lags:
        if 0 <= lag < len(cc):
            corr_results.append([topic, lag, cc[lag]])

pd.DataFrame(
    corr_results,
    columns=["topic_id", "lag_weeks", "correlation"]
).to_csv("output_topic_case_lag_corr.csv", index=False)

print("Lag correlation saved.")

# Plot stage timeline with case count
plt.figure(figsize=(12, 4))
sns.lineplot(data=covid, x="week", y="daily_new_cases", hue="stage", palette="tab10")
plt.title("COVID Cases with Stage Segmentation")
plt.savefig("plot_stage_cases.png", dpi=300)
plt.close()

print("Stage analysis completed successfully.")

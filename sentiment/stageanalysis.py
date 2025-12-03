#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage_analysis.py
疫情阶段对比分析（Stage Comparison）
功能：
1. 自动疫情阶段划分（ruptures）
2. 各阶段 Top 主题变化
3. 情绪均值显著性检验（t-test / KS）
4. 主题热度与新增病例滞后相关（cross-correlation）
5. 多个可视化图表输出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
from scipy.stats import ttest_ind, ks_2samp
from statsmodels.tsa.stattools import ccf

# ============================================================
# 1. 读取数据
# ============================================================
print("📌 Loading sentiment & topic data...")
df = pd.read_csv("tweets_with_sentiment.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df["week"] = df["Datetime"].dt.to_period("W").dt.start_time

print("📌 Loading COVID weekly data...")
covid = pd.read_csv("clean_covid_weekly.csv")
covid["week"] = pd.to_datetime(covid["week"], errors="coerce")

print("✔ Data loaded.")

# ============================================================
# 2. 自动疫情阶段划分（ruptures）
# ============================================================
print("\n🚀 Running change point detection (ruptures)...")

series = covid["daily_new_cases"].values
algo = rpt.Pelt(model="rbf").fit(series)

# 自动选择 4 个阶段（你可以调）
breaks = algo.predict(pen=5)
breaks = sorted(list(set(breaks)))

print("阶段分割点（按周序号）：", breaks)

covid["stage"] = 0
for i in range(len(breaks) - 1):
    covid.loc[breaks[i]:breaks[i+1], "stage"] = i + 1

print("✔ Stage segmentation completed.")

# ============================================================
# 3. 合并推文到疫情阶段
# ============================================================
df = df.merge(covid[["week", "stage", "daily_new_cases"]], on="week", how="left")

print("✔ Tweets matched to COVID stages.")


# ============================================================
# 4. 各阶段主题热度
# ============================================================
print("\n📊 Computing stage-based topic frequencies...")

topic_freq = df.groupby(["stage", "topic_id"]).size().reset_index(name="count")
topic_freq.to_csv("output_stage_topic_frequency.csv", index=False)

# Top 10 主题可视化
plt.figure(figsize=(12, 6))
sns.barplot(data=topic_freq, x="stage", y="count", hue="topic_id")
plt.title("Topic Frequency per Stage")
plt.xlabel("Stage")
plt.ylabel("Tweet Count")
plt.legend(title="Topic ID")
plt.tight_layout()
plt.savefig("plot_stage_topic_freq.png", dpi=300)
plt.close()


# ============================================================
# 5. 情绪显著性检验（t-test & KS）
# ============================================================
print("\n🔍 Performing emotion significance tests...")

emotions = ["sent_pos", "sent_neu", "sent_neg"]

test_results = []

stages = sorted(df["stage"].unique())

for e in emotions:
    for i in range(len(stages)-1):
        s1 = df[df["stage"] == stages[i]][e].dropna()
        s2 = df[df["stage"] == stages[i+1]][e].dropna()

        t_p = ttest_ind(s1, s2, equal_var=False).pvalue
        ks_p = ks_2samp(s1, s2).pvalue

        test_results.append([e, stages[i], stages[i+1], t_p, ks_p])

test_df = pd.DataFrame(test_results, columns=[
    "emotion", "stage1", "stage2", "t_test_p", "ks_test_p"
])
test_df.to_csv("output_emotion_significance.csv", index=False)

print("✔ Emotion significance tests completed.")


# ============================================================
# 6. 滞后相关（Cross-correlation）
# ============================================================
print("\n📈 Computing cross-correlation (topic vs new cases)...")

# 每周每主题出现次数
weekly_topic = df.groupby(["week", "topic_id"]).size().unstack(fill_value=0)

# 对齐病例数据
aligned = covid.set_index("week")[["daily_new_cases"]].join(weekly_topic, how="left").fillna(0)

lags = range(-4, 5)  # 从 -4 到 +4 周滞后
corr_results = []

for topic in weekly_topic.columns:
    t_series = aligned[topic].values
    cases_series = aligned["daily_new_cases"].values

    cc = ccf(t_series, cases_series)

    for lag in lags:
        if 0 <= lag < len(cc):
            corr_results.append([topic, lag, cc[lag]])

corr_df = pd.DataFrame(corr_results, columns=["topic_id", "lag_weeks", "correlation"])
corr_df.to_csv("output_topic_case_lag_corr.csv", index=False)

print("✔ Cross-correlation completed.")


# ============================================================
# 7. 可视化：疫情阶段热力图
# ============================================================
plt.figure(figsize=(10, 4))
sns.lineplot(data=covid, x="week", y="daily_new_cases", hue="stage", palette="tab10")
plt.title("COVID Cases with Stage Segmentation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_stage_cases.png", dpi=300)
plt.close()

print("\n🎉 All stage-comparison analysis completed!")
print("生成文件包括：")
print("- plot_stage_topic_freq.png")
print("- output_stage_topic_frequency.csv")
print("- output_emotion_significance.csv")
print("- output_topic_case_lag_corr.csv")
print("- plot_stage_cases.png")

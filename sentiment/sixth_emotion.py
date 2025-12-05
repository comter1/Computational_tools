"""
Enhanced Emotion Time Series Visualization

Plots generated:
1) Monthly neutral vs non-neutral trend
2) Monthly Top-5 fluctuating emotions (normalized, neutral removed)
3) Monthly emotion ratio heatmap (normalized, non-neutral only)
4) Monthly dominant emotion timeline
5) Stage-based emotion ratio heatmap (P1–P4)
6) Radar chart comparing emotional distributions between stages
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load sentiment data generated from previous pipeline
df = pd.read_csv("tweets_with_sentiment.csv")

# Convert datetime format
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

# Filter for valid study time range
df = df[(df["Datetime"] >= "2020-01-01") & (df["Datetime"] <= "2021-02-28")]
print(f"Loaded {len(df)} tweets within 2020–2021 window.")

# Extract monthly timestamp
df["month"] = df["Datetime"].dt.to_period("M").dt.start_time


# 1. Neutral vs Non-neutral Emotion Trend
df["non_neutral"] = 1 - df["emo_neutral"]
monthly_nn = df.groupby("month")[["emo_neutral", "non_neutral"]].mean()

plt.figure(figsize=(12, 5))
plt.plot(monthly_nn.index, monthly_nn["emo_neutral"], label="Neutral", linewidth=2)
plt.plot(monthly_nn.index, monthly_nn["non_neutral"], label="Non-neutral", linewidth=2)
plt.title("Monthly Neutral vs Non-neutral (GoEmotions)")
plt.xlabel("Month")
plt.ylabel("Average Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_monthly_neutral_vs_nonneutral.png", dpi=300)
plt.close()

print("Saved: plot_monthly_neutral_vs_nonneutral.png")


# 2. Monthly Top-5 Non-neutral Fluctuating Emotions
emotion_cols = [c for c in df.columns if c.startswith("emo_")]
emotion_cols_no_neutral = [c for c in emotion_cols if c != "emo_neutral"]

monthly_non = df.groupby("month")[emotion_cols_no_neutral].mean()
monthly_non_norm = monthly_non.div(monthly_non.sum(axis=1), axis=0)

var_norm = monthly_non_norm.var().sort_values(ascending=False)
top5_norm = var_norm.head(5).index.tolist()

plt.figure(figsize=(12,6))
for emo in top5_norm:
    plt.plot(monthly_non_norm.index, monthly_non_norm[emo], label=emo, linewidth=2)

plt.title("Monthly Normalized Top-5 Emotions (Neutral Removed)")
plt.xlabel("Month")
plt.ylabel("Normalized Probability Share")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_top5_normalized_emotions.png", dpi=300)
plt.close()

print("Saved: plot_top5_normalized_emotions.png")


# 3. Monthly Emotion Ratio Heatmap (Without Neutral)
ratio = df.groupby("month")[emotion_cols_no_neutral].sum()
ratio = ratio.div(ratio.sum(axis=1), axis=0)

plt.figure(figsize=(14, 10))
sns.heatmap(ratio.T, cmap="coolwarm", linewidths=0.3)
plt.title("Monthly Emotion Ratio Heatmap (Normalized Non-neutral)")
plt.xlabel("Month")
plt.ylabel("Emotion")
plt.tight_layout()
plt.savefig("plot_emotion_ratio_heatmap.png", dpi=300)
plt.close()

print("Saved: plot_emotion_ratio_heatmap.png")


# 4. Monthly Dominant Emotion Timeline
monthly_major = df.groupby("month")["main_emotion"].agg(lambda x: x.value_counts().index[0])

plt.figure(figsize=(12, 5))
plt.plot(monthly_major.index, monthly_major.values, marker="o", linestyle="-", linewidth=2)
plt.title("Monthly Dominant Emotion (GoEmotions)")
plt.xlabel("Month")
plt.ylabel("Dominant Emotion")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_monthly_dominant_emotion.png", dpi=300)
plt.close()

print("Saved: plot_monthly_dominant_emotion.png")


# 5. Stage-based Heatmap of Emotion Distribution
print("\nGenerating stage-based heatmap...")

stages = [
    ("P1_early",          "2020-01-01", "2020-02-29"),
    ("P2_surge",          "2020-03-01", "2020-05-31"),
    ("P3_summer_relief",  "2020-06-01", "2020-09-30"),
    ("P4_second_wave",    "2020-10-01", "2021-02-28"),
]

stage_results = {}

for stage_name, start, end in stages:
    df_stage = df[(df["Datetime"] >= start) & (df["Datetime"] <= end)]
    emo_sum = df_stage[emotion_cols_no_neutral].sum()
    emo_ratio = emo_sum / emo_sum.sum()
    stage_results[stage_name] = emo_ratio

stage_df = pd.DataFrame(stage_results)

plt.figure(figsize=(10, 12))
sns.heatmap(stage_df, cmap="coolwarm", linewidths=0.5)
plt.title("Stage-based Emotion Ratio Heatmap (Non-neutral Normalized)")
plt.xlabel("Stage")
plt.ylabel("Emotion")
plt.tight_layout()
plt.savefig("plot_stage_emotion_ratio_heatmap.png", dpi=300)
plt.show()

print("Saved: plot_stage_emotion_ratio_heatmap.png")


# 6. Radar Chart Comparison Between Stages
print("\nGenerating radar charts for stages...")

from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes

overall_mean = stage_df.mean(axis=1)
top_emotions = overall_mean.sort_values(ascending=False).head(8).index.tolist()

def radar_factory(num_vars):
    """Returns evenly spaced angles around a circle."""
    return np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

class RadarAxes(PolarAxes):
    name = 'radar'
    def fill(self, *args, closed=True, **kwargs):
        return super().fill(closed=closed, *args, **kwargs)
    def plot(self, *args, **kwargs):
        lines = super().plot(*args, **kwargs)
        for line in lines:
            line.set_clip_on(False)
        return lines
    def set_varlabels(self, labels):
        self.set_thetagrids(np.degrees(theta), labels)

register_projection(RadarAxes)

theta = radar_factory(len(top_emotions))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="radar"))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for idx, (stage_name, _, _) in enumerate(stages):
    values = stage_df.loc[top_emotions, stage_name].values
    ax.plot(theta, values, label=stage_name, color=colors[idx], linewidth=2)
    ax.fill(theta, values, alpha=0.25, color=colors[idx])

ax.set_varlabels(top_emotions)
plt.title("Radar Chart of Top Emotions Across Stages (Non-neutral)", fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("plot_stage_emotion_radar.png", dpi=300)
plt.show()

print("Saved: plot_stage_emotion_radar.png")
print("Radar charts generated successfully.")

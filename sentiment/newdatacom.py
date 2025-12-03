#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
covid_preprocess.py
将 JHU COVID-19 Daily Report（省级）转为：
1）全国累计确诊
2）全国每日新增
3）全国每周新增
输出：clean_covid_timeseries.csv
"""

import pandas as pd

INPUT_FILE = "covid_19_data.csv"   # 你的疫情数据文件名

# ============================================================
# 1. 加载数据
# ============================================================
df = pd.read_csv(INPUT_FILE)

# 转换日期格式
df["Observation Date"] = pd.to_datetime(df["Observation Date"], errors="coerce")

# ============================================================
# 2. 按日期聚合：全国累计确诊
# ============================================================
national = df.groupby("Observation Date")["Confirmed"].sum().reset_index()
national = national.sort_values("Observation Date")

# ============================================================
# 3. 计算每日新增
# ============================================================
national["daily_new_cases"] = national["Confirmed"].diff().fillna(0)

# 去掉负值（数据修正导致的）
national["daily_new_cases"] = national["daily_new_cases"].clip(lower=0)

# ============================================================
# 4. 按周聚合新增病例
# ============================================================
national["week"] = national["Observation Date"].dt.to_period("W").dt.start_time

weekly = national.groupby("week")["daily_new_cases"].sum().reset_index()

# ============================================================
# 5. 保存结果
# ============================================================
national.to_csv("clean_covid_timeseries.csv", index=False, encoding="utf-8")
weekly.to_csv("clean_covid_weekly.csv", index=False, encoding="utf-8")

print("✔ Saved national daily data → clean_covid_timeseries.csv")
print("✔ Saved weekly aggregated data → clean_covid_weekly.csv")
print(national.head())

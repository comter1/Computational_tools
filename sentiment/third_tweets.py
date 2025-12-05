#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 00:59:43 2025

@author: liuxiaosa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 22:22:58 2025

@author: liuxiaosa
"""

"""
process_clean_text.py
Load raw crawler CSV files → clean text → output processed_tweets.csv
"""

import re
import pandas as pd
import glob
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Text cleaning function
def clean_text(text):
    """
    Clean raw tweet text. Operations include:
        - Remove URLs
        - Remove @mentions
        - Remove hashtag symbol (#)
        - Remove email addresses
        - Remove emojis
        - Convert to lowercase
        - Remove punctuation
        - Remove extra whitespace
    """
    if text is None:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove @username mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol (#)
    text = text.replace("#", "")

    # Remove email patterns
    text = re.sub(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", "", text)

    # Remove emoji characters
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+", 
        flags=re.UNICODE)
    text = emoji_pattern.sub("", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load all CSV files matching a pattern
def load_all_csv(pattern="data/2020-*.csv"):
    """
    Load multiple CSV files using a pattern.
    Return a single concatenated DataFrame.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No CSV files found under the given pattern.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# Main script
if __name__ == "__main__":
    print("Loading raw tweet CSV files...")
    df = load_all_csv("data/202*-*.csv")

    # Ensure required column exists
    if "Content" not in df.columns:
        raise ValueError("The input CSV is missing the 'Content' column (tweet text).")

    print("Cleaning text...")
    df["clean_text"] = df["Content"].astype(str).apply(clean_text)

    print("Saving to processed_tweets.csv...")
    df.to_csv("processed_tweets.csv", index=False, encoding="utf-8")

    print("Task completed. Output file: processed_tweets.csv")

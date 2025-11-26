import pandas as pd
import re

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def make_clean_csv():
    df = pd.read_csv("app_src/data/processed/amazon_3k.csv")

    df["clean_text"] = df["review_text"].astype(str).apply(clean_text)
    df["sentiment"] = df["label"]

    df.to_csv("app_src/data/processed/amazon_3k.csv", index=False)
    print("✔ clean file saved as amazon_10k.csv")

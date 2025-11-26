import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE, "data/processed/amazon_3k.csv")  # << 3k dataset
SAVE_PATH = os.path.join(BASE, "models/sentiment_model")

os.makedirs(SAVE_PATH, exist_ok=True)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_sentiment():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    texts = df["clean_text"].tolist()
    labels = df["sentiment"].astype(int).tolist()

    print("🔠 Loading DistilBERT (FAST)...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )

    dataset = ReviewDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-5)

    print("🚀 Training started (FAST MODE)...")
    model.train()

    for epoch in range(1):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    print("💾 Saving sentiment model...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("✅ FAST Sentiment Model Training Completed!")

if __name__ == "__main__":
    train_sentiment()

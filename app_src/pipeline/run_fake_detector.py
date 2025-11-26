import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE, "data/processed/amazon_3k.csv")

SAVE_MODEL = os.path.join(BASE, "models/fake_review_model.pkl")
SAVE_VEC = os.path.join(BASE, "models/fake_vectorizer.pkl")

def train_fake_review():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df["clean_text"]
    y = df["label"]

    print("🔠 Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    print("🌲 Training RandomForest...")
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X_vec, y)

    print("💾 Saving model & vectorizer...")
    joblib.dump(model, SAVE_MODEL)
    joblib.dump(vectorizer, SAVE_VEC)

    print("✅ Fake Review Model Training Completed!")

if __name__ == "__main__":
    train_fake_review()

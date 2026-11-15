
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# =====================================
# 1. Load Dataset
# =====================================

print("Loading dataset...")

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")


# =====================================
# 2. Create Labels
# =====================================

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], ignore_index=True)

print("Original dataset:", df.shape)


# =====================================
# 3. Data Preprocessing
# =====================================

df = df.dropna(subset=["title", "text"])

df["title"] = df["title"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()

df = df[
    (df["title"] != "") &
    (df["text"] != "")
]

# Remove exact duplicate news within each label
df = df.drop_duplicates(
    subset=["title", "text", "label"]
)

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("Cleaned dataset:", df.shape)


# =====================================
# 4. Split Dataset
# =====================================

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================
# 5. Build TF-IDF + Logistic Regression
# =====================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_df=0.7,
            max_features=50000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# =====================================
# 6. Train Model
# =====================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# =====================================
# 7. Make Predictions
# =====================================

y_pred = model.predict(X_test)


# =====================================
# 8. Evaluate Model
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== Model Evaluation =====")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake News", "Real News"]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(y_test, y_pred)
)


# =====================================
# 9. Save Model
# =====================================

import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/fake_news_model.pkl"
)

print("\nModel saved successfully!")

print("\nFake News Detection Project Completed!")

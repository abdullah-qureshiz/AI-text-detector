"""train.py - trains the machine-learning detector.

Run:  python train.py
Needs a CSV file with two columns:  text,label   (label: 0 = human, 1 = AI)
"""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import Config


class ModelTrainer:
    def __init__(self, data_path=Config.DATA_PATH, model_path=Config.MODEL_PATH):
        self.data_path = data_path
        self.model_path = model_path
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                      min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text")
        print(f"Loaded {len(df)} rows. Label counts:\n{df['label'].value_counts().to_string()}")
        if len(df) < 1000:
            print("\nWARNING: very small dataset. Scores will look better than they really are.")
            print("Use a real dataset (thousands of rows) for meaningful results.\n")
        return df

    def split(self, df):
        # stratify keeps the human/AI ratio the same in train and test
        return train_test_split(df["text"], df["label"], test_size=0.2,
                                random_state=42, stratify=df["label"])

    def train(self):
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.split(df)
        self.pipeline.fit(X_train, y_train)            # TF-IDF is learned on TRAIN data only
        self.evaluate(X_test, y_test)
        self.save()

    def evaluate(self, X_test, y_test):
        predictions = self.pipeline.predict(X_test)
        print("Classification report (0 = human, 1 = AI):")
        print(classification_report(y_test, predictions,
                                    target_names=["human", "ai"], zero_division=0))
        print("Confusion matrix [rows = real, columns = predicted]:")
        print(confusion_matrix(y_test, predictions))

    def save(self):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"\nModel saved to {self.model_path}")


if __name__ == "__main__":
    ModelTrainer().train()
"""detectors.py - all detector classes.

BaseTextDetector      (parent)
 |-- RuleBasedDetector  (child: simple statistics)
 `-- MLTextDetector     (child: trained scikit-learn model)
"""
import os
import re
import statistics

import joblib

from config import Config


class TextPreprocessor:
    """Cleans text and splits it into sentences and words."""

    @staticmethod
    def clean(text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)      # many spaces -> one space
        return text

    @staticmethod
    def sentences(text):
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def words(text):
        return re.findall(r"[a-zA-Z']+", text.lower())


class BaseTextDetector:
    """Parent class. Shared code lives here; child classes override predict()."""

    name = "Base detector"

    def __init__(self, threshold=Config.THRESHOLD):
        self.threshold = threshold

    # ---- shared by all children (inheritance) ----
    def validate(self, text):
        """Return an error message, or None if the text is fine."""
        if not text or not text.strip():
            return "Please enter some text."
        if len(text) > Config.MAX_CHARS:
            return f"Text is too long (max {Config.MAX_CHARS} characters)."
        word_count = len(TextPreprocessor.words(text))
        if word_count < Config.MIN_WORDS:
            return f"Please enter at least {Config.MIN_WORDS} words (you gave {word_count})."
        return None

    def interpret(self, ai_probability):
        """Turn a probability (0..1) into a label and a confidence percentage."""
        if ai_probability >= self.threshold:
            label, confidence = "AI-Generated", ai_probability
        else:
            label, confidence = "Human-Written", 1 - ai_probability
        return {
            "detector": self.name,
            "label": label,
            "confidence": round(confidence * 100, 1),
            "ai_probability": round(ai_probability, 3),
        }

    def analyze(self, text):
        """Public method used by main.py: validate -> predict -> interpret."""
        error = self.validate(text)
        if error:
            return {"error": error}
        text = TextPreprocessor.clean(text)
        ai_probability, details = self.predict(text)   # <- child's own version runs here
        result = self.interpret(ai_probability)
        result["details"] = details
        result["note"] = ("This is an estimate, not proof. Detectors can be wrong, "
                          "especially for short text or non-native English writers.")
        return result

    # ---- must be overridden ----
    def predict(self, text):
        """Return (ai_probability, details_dict). Children must override this."""
        raise NotImplementedError("Child classes must override predict()")


class RuleBasedDetector(BaseTextDetector):
    """Child 1: looks at simple writing statistics (no training needed).

    Idea: AI text is often very even (similar sentence lengths) and uses
    a smoother, more repeated vocabulary. These are weak signals only.
    """

    name = "Rule-based detector"

    def predict(self, text):                      # method overriding
        sentences = TextPreprocessor.sentences(text)
        words = TextPreprocessor.words(text)
        lengths = [len(TextPreprocessor.words(s)) for s in sentences] or [0]

        mean_len = statistics.mean(lengths)
        std_len = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        burstiness = std_len / mean_len if mean_len else 0.0       # low = very even
        diversity = len(set(words)) / len(words) if words else 0.0  # low = repetitive

        # Convert each feature to a 0..1 "AI-like" score
        even_score = max(0.0, min(1.0, 1 - burstiness / 0.8))
        repeat_score = max(0.0, min(1.0, (0.75 - diversity) / 0.35))

        ai_probability = 0.6 * even_score + 0.4 * repeat_score
        details = {
            "sentences": len(sentences),
            "avg_sentence_length": round(mean_len, 1),
            "burstiness": round(burstiness, 2),
            "vocabulary_diversity": round(diversity, 2),
        }
        return ai_probability, details


class MLTextDetector(BaseTextDetector):
    """Child 2: uses a trained TF-IDF + Logistic Regression model."""

    name = "Machine-learning detector"

    def __init__(self, model_path=Config.MODEL_PATH, threshold=Config.THRESHOLD):
        super().__init__(threshold)               # run the parent's __init__ first
        self.model_path = model_path
        self.model = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. Run: python train.py")
        self.model = joblib.load(self.model_path)
        return self

    def predict(self, text):                      # method overriding
        if self.model is None:
            self.load()
        ai_probability = float(self.model.predict_proba([text])[0][1])  # class 1 = AI
        return ai_probability, {"model": "TF-IDF + Logistic Regression"}
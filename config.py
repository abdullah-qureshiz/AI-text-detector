"""config.py - loads settings from the .env file in one place."""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env and puts the values into environment variables


class Config:
    DATA_PATH = os.getenv("DATA_PATH", "data/sample_data.csv")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/text_model.joblib")
    MIN_WORDS = int(os.getenv("MIN_WORDS", "30"))
    MAX_CHARS = int(os.getenv("MAX_CHARS", "20000"))
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
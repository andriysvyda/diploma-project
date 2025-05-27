import re
import pandas as pd
from transformers import pipeline
import spacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("uk_core_news_sm")

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+|\@\w+', '', text)
    text = re.sub(r'[^\w\s\u0400-\u04FF]', '', text)
    text = text.lower()
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return ' '.join(lemmatized_words)

logger.info("Початок класифікації...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
topics = ["політика", "економіка", "технології", "культура", "спорт"]

df = pd.read_csv("tweets_ukraine.csv")
logger.info(f"Завантажено {len(df)} твітів")
df["text"] = df["text"].apply(preprocess_text)
df["topic"] = df["text"].apply(lambda x: classifier(x, topics)["labels"][0] if x.strip() else "невідомо")
df.to_csv("classified_tweets.csv", index=False)
logger.info("Класифікація завершена")
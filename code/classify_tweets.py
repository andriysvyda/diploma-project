import re
import pandas as pd
from transformers import pipeline
import pymorphy2
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ініціалізація лематизатора
morph = pymorphy2.MorphAnalyzer(lang='uk')

def preprocess_text(text):
    """Очищення та лематизація тексту"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)

logger.info("Початок класифікації...")
classifier = pipeline("zero-shot-classification", model="facebook/mbart-large-50")
topics = ["політика", "економіка", "технології", "культура", "спорт"]

df = pd.read_csv("tweets_ukraine.csv")
logger.info(f"Завантажено {len(df)} твітів")
df["text"] = df["text"].apply(preprocess_text)
df["topic"] = df["text"].apply(lambda x: classifier(x, topics)["labels"][0])
df.to_csv("classified_tweets.csv", index=False)
logger.info("Класифікація завершена")
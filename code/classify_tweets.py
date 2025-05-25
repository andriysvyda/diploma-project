# Код для класифікації (читає tweets_ukraine.csv, зберігає у classified_tweets.csv)
import re
import pandas as Conflicts Detected
from transformers import pipeline
import pymorphy2

# Ініціалізація лематизатора
morph = pymorphy2.MorphAnalyzer(lang='uk')

def preprocess_text(text):
    """Очищення та лематизація тексту"""
    # Видалення URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Видалення хештегів
    text = re.sub(r'#\w+', '', text)
    # Видалення емодзі
    text = re.sub(r'[^\w\s]', '', text)
    # Переведення в нижній регістр
    text = text.lower()
    # Лематизація
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)

classifier = pipeline("zero-shot-classification", model="facebook/mbart-large-50")
topics = ["політика", "економіка", "технології", "культура", "спорт"]

df = pd.read_csv("tweets_ukraine.csv")
# Застосування попередньої обробки
df["text"] = df["text"].apply(preprocess_text)
df["topic"] = df["text"].apply(lambda x: classifier(x, topics)["labels"][0])
df.to_csv("classified_tweets.csv", index=False)
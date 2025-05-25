# Код для класифікації (читає tweets_ukraine.csv, зберігає у classified_tweets.csv)
from transformers import pipeline
import pandas as pd

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
topics = ["політика", "економіка", "технології", "культура", "спорт"]

df = pd.read_csv("tweets_ukraine.csv")
df["topic"] = df["text"].apply(lambda x: classifier(x, topics)["labels"][0])
df.to_csv("classified_tweets.csv", index=False)
# Код для графіка (читає classified_tweets.csv, зберігає зображення)
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("classified_tweets.csv")
df["topic"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Розподіл тем твітів")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("topics_plot.png")  # Зберігає графік у PNG
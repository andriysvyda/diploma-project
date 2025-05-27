import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_topic_distribution():
    try:
        df = pd.read_csv("classified_tweets.csv")
        logger.info(f"Завантажено {len(df)} твітів для побудови графіку.")
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x="topic", order=df["topic"].value_counts().index)
        plt.title("Розподіл твітів за темами")
        plt.xlabel("Тема")
        plt.ylabel("Кількість твітів")
        plt.xticks(rotation=45)
        
        output_path = "topics_plot.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Графік збережено у {output_path}")
    except Exception as e:
        logger.error(f"Помилка при побудові графіку: {str(e)}")

if __name__ == "__main__":
    plot_topic_distribution()
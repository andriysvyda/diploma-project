import torch
from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path="data/tweets_ukraine.csv", sample_size=20):
    """Завантажує твіти з CSV"""
    try:
        df = pd.read_csv(file_path)
        texts = df["text"].tolist()[:sample_size]
        logger.info(f"Успішно завантажено {len(texts)} твітів.")
        return texts
    except Exception as e:
        logger.error(f"Помилка завантаження даних: {e}")
        raise

def classify_texts(texts, topics, model_name="facebook/bart-large-mnli"):
    """Класифікує тексти за допомогою моделі"""
    try:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        
        predictions = []
        for text in tqdm(texts, desc="Класифікація"):
            try:
                result = classifier(text, topics)
                predictions.append(result["labels"][0])
            except Exception as e:
                logger.warning(f"Помилка класифікації тексту: {text[:50]}...")
                predictions.append("невідомо")
        
        return predictions
    except Exception as e:
        logger.error(f"Помилка ініціалізації моделі: {e}")
        raise

def evaluate_results(true_topics, predicted_topics):
    """Оцінює точність моделі"""
    if not true_topics or len(true_topics) != len(predicted_topics):
        logger.warning("Немає даних для оцінки")
        return None
    
    accuracy = accuracy_score(true_topics, predicted_topics)
    logger.info(f"Точність моделі: {accuracy:.2%}")
    
    # Додаткові метрики
    from sklearn.metrics import classification_report
    print("\nДетальний звіт:")
    print(classification_report(true_topics, predicted_topics))
    
    return accuracy

def save_results(texts, predictions, true_topics=None, output_file="classification_results.csv"):
    """Зберігає результати"""
    results = {
        "text": texts,
        "predicted_topic": predictions,
    }
    
    if true_topics and len(true_topics) >= len(texts):
        results["true_topic"] = true_topics[:len(texts)]
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.info(f"Результати збережено у {output_file}")

if __name__ == "__main__":
    # Конфігурація
    TOPICS = ["політика", "економіка", "технології", "культура", "спорт"]
    MANUAL_LABELS = ["політика"] * 5 + ["економіка"] * 5 + ["технології"] * 5 + ["культура"] * 5
    
    try:
        # 1. Завантаження даних
        tweets = load_data(sample_size=20)
        
        # 2. Класифікація
        predictions = classify_texts(tweets, TOPICS)
        
        # 3. Оцінка
        if MANUAL_LABELS:
            evaluate_results(MANUAL_LABELS[:len(tweets)], predictions)
        
        # 4. Збереження
        save_results(tweets, predictions, MANUAL_LABELS)
        
    except Exception as e:
        logger.critical(f"Критична помилка: {e}")
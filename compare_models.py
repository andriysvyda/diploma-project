import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path="classified_tweets.csv", sample_size=20):
    try:
        df = pd.read_csv(file_path)
        texts = df["text"].tolist()[:sample_size]
        true_topics = df["topic"].tolist()[:sample_size]
        logger.info(f"Успішно завантажено {len(texts)} твітів.")
        return texts, true_topics
    except Exception as e:
        logger.error(f"Помилка завантаження даних: {e}")
        raise

def classify_texts_bart(texts, topics):
    try:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
        predictions = []
        for text in tqdm(texts, desc="Класифікація BART"):
            try:
                result = classifier(text, topics)
                predictions.append(result["labels"][0])
            except Exception as e:
                logger.warning(f"Помилка класифікації тексту: {text[:50]}...")
                predictions.append("невідомо")
        return predictions
    except Exception as e:
        logger.error(f"Помилка ініціалізації моделі BART: {e}")
        raise

def classify_texts_roberta(texts, topics, model_path="roberta_training/checkpoint-3"):
    try:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        predictions = []
        label_map = {i: t for i, t in enumerate(topics)}
        for text in tqdm(texts, desc="Класифікація RoBERTa"):
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            pred_label = outputs.logits.argmax(-1).item()
            predictions.append(label_map[pred_label])
        return predictions
    except Exception as e:
        logger.error(f"Помилка ініціалізації моделі RoBERTa: {e}")
        raise

def evaluate_results(true_topics, predicted_topics, model_name):
    if not true_topics or len(true_topics) != len(predicted_topics):
        logger.warning(f"Немає даних для оцінки {model_name}")
        return None
    
    accuracy = accuracy_score(true_topics, predicted_topics)
    f1 = f1_score(true_topics, predicted_topics, average="weighted")
    logger.info(f"{model_name} - Точність: {accuracy:.2%}, F1-score: {f1:.2%}")
    
    report = classification_report(true_topics, predicted_topics)
    print(f"\nДетальний звіт для {model_name}:\n{report}")
    
    return {"accuracy": accuracy, "f1": f1}

def save_results(texts, bart_predictions, roberta_predictions, true_topics, output_file="classification_results.csv"):
    results = {
        "text": texts,
        "bart_predicted_topic": bart_predictions,
        "roberta_predicted_topic": roberta_predictions,
        "true_topic": true_topics
    }
    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.info(f"Результати збережено у {output_file}")

if __name__ == "__main__":
    TOPICS = ["політика", "економіка", "технології", "культура", "спорт"]
    
    try:
        # Завантаження даних
        tweets, true_topics = load_data()
        
        # Класифікація BART
        bart_predictions = classify_texts_bart(tweets, TOPICS)
        
        # Класифікація RoBERTa
        roberta_predictions = classify_texts_roberta(tweets, TOPICS)
        
        # Оцінка
        bart_metrics = evaluate_results(true_topics, bart_predictions, "BART")
        roberta_metrics = evaluate_results(true_topics, roberta_predictions, "RoBERTa")
        
        # Збереження
        save_results(tweets, bart_predictions, roberta_predictions, true_topics)
        
    except Exception as e:
        logger.critical(f"Критична помилка: {e}")
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path="data/tweets_ukraine.csv", sample_size=20):
    try:
        df = pd.read_csv(file_path)
        texts = df["text"].tolist()[:sample_size]
        logger.info(f"Успішно завантажено {len(texts)} твітів.")
        return texts
    except Exception as e:
        logger.error(f"Помилка завантаження даних: {e}")
        raise

def classify_texts_bart(texts, topics, model_name="facebook/mbart-large-50"):
    try:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)
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

def classify_texts_roberta(texts, topics, model_path="./results/checkpoint-best"):
    try:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        predictions = []
        label_map = {i: t for i, t in enumerate(topics)}  # Зворотне відображення
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
    
    from sklearn.metrics import classification_report
    print(f"\nДетальний звіт для {model_name}:")
    print(classification_report(true_topics, predicted_topics))
    
    return {"accuracy": accuracy, "f1": f1}

def save_results(texts, bart_predictions, roberta_predictions, true_topics=None, output_file="classification_results.csv"):
    results = {
        "text": texts,
        "bart_predicted_topic": bart_predictions,
        "roberta_predicted_topic": roberta_predictions,
    }
    if true_topics and len(true_topics) >= len(texts):
        results["true_topic"] = true_topics[:len(texts)]
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    logger.info(f"Результати збережено у {output_file}")

if __name__ == "__main__":
    TOPICS = ["політика", "економіка", "технології", "культура", "спорт"]
    # Оновлені ручні мітки (по 4 для кожної теми)
    MANUAL_LABELS = ["політика"] * 4 + ["економіка"] * 4 + ["технології"] * 4 + ["культура"] * 4 + ["спорт"] * 4
    
    try:
        # Завантаження даних
        tweets = load_data(sample_size=20)
        
        # Класифікація BART
        bart_predictions = classify_texts_bart(tweets, TOPICS)
        
        # Класифікація RoBERTa
        roberta_predictions = classify_texts_roberta(tweets, TOPICS)
        
        # Оцінка
        if MANUAL_LABELS:
            bart_metrics = evaluate_results(MANUAL_LABELS[:len(tweets)], bart_predictions, "BART")
            roberta_metrics = evaluate_results(MANUAL_LABELS[:len(tweets)], roberta_predictions, "RoBERTa")
        
        # Збереження
        save_results(tweets, bart_predictions, roberta_predictions, MANUAL_LABELS)
        
    except Exception as e:
        logger.critical(f"Критична помилка: {e}")
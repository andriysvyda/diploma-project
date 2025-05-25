# train_roberta.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import pymorphy2
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

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

def load_data(file_path="classified_tweets.csv"):
    df = pd.read_csv(file_path)
    df['text'] = df['text'].astype(str).apply(preprocess_text)
    texts = df['text'].tolist()
    topics = df['topic'].tolist()
    unique_topics = list(set(topics))
    label_map = {t: i for i, t in enumerate(unique_topics)}
    labels = [label_map[t] for t in topics]
    return texts, labels, label_map

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model():
    try:
        logger.info("Початок тренування...")
        texts, labels, label_map = load_data()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Використання мультимовної моделі
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(label_map)
        )
        
        train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)
        val_dataset = SimpleDataset(val_texts, val_labels, tokenizer)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,  # Збільшено для кращого навчання
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",  # Оцінка після кожної епохи
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda eval_pred: {
                "accuracy": accuracy_score(eval_pred.label_ids, eval_pred.predictions.argmax(-1)),
                "f1": f1_score(eval_pred.label_ids, eval_pred.predictions.argmax(-1), average="weighted")
            }
        )
        trainer.train()
        
        logger.info("Тренування завершено!")
        
    except Exception as e:
        logger.error(f"Помилка: {str(e)}")

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, f1_score
    train_model()
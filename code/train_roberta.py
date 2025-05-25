# train_roberta.py - максимально спрощений робочий варіант
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

def load_data(file_path="classified_tweets.csv"):
    df = pd.read_csv(file_path)
    texts = df['text'].astype(str).tolist()
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
        
        # 1. Завантаження даних
        texts, labels, label_map = load_data()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 2. Завантаження моделі
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=len(label_map)
        )
        
        # 3. Підготовка даних
        train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)
        
        # 4. Налаштування тренування
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,  # Зменшено для тесту
            per_device_train_batch_size=4,  # Зменшено для тесту
            save_strategy='no'  # Вимкнено збереження проміжних результатів
        )
        
        # 5. Тренування
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )
        trainer.train()
        
        logger.info("Тестове тренування завершено!")
        
    except Exception as e:
        logger.error(f"Помилка: {str(e)}")
        logger.info("\nЯкщо виникає помилка з accelerate, спробуйте:")
        logger.info("1. pip install 'accelerate>=0.26.0'")
        logger.info("2. pip install transformers[torch]")
        logger.info("3. Перезапустіть середовище Python")

if __name__ == "__main__":
    train_model()
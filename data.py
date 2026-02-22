import requests
import gzip
import json
import random
import torch
from utils import genre_url_dict, label2id

def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
                
    return random.sample(reviews, min(sample_size, len(reviews)))

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_data(tokenizer, max_length=512):
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, url in genre_url_dict.items():
        # Using 1000 sample size as demonstrated in the notebook
        reviews = load_reviews(url, head=10000, sample_size=1000)
        
        for review in reviews[:800]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in reviews[800:]:
            test_texts.append(review)
            test_labels.append(genre)

    # Encode texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    # Encode labels
    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    # Create PyTorch datasets
    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    return train_dataset, test_dataset
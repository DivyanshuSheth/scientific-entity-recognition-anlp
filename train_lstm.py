# The "AnnotatedData" folder contains NER data in CoNLL format.
# Read from that folder and train a huggingface AutoModelForTokenClassification model.
# Print training metrics and performance of model on each class

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support
seqeval = evaluate.load("seqeval")

label_to_idx = {"O": 0,
           "B-MethodName": 1,
           "I-MethodName": 2,
           "B-HyperparameterName": 3,
           "I-HyperparameterName": 4,
           "B-HyperparameterValue": 5,
           "I-HyperparameterValue": 6,
           "B-MetricName": 7,
           "I-MetricName": 8,
           "B-MetricValue": 9,
           "I-MetricValue": 10,
           "B-TaskName": 11,
           "I-TaskName": 12,
           "B-DatasetName": 13,
           "I-DatasetName": 14,
           }
idx_to_label = {v: k for k, v in label_to_idx.items()}
label_list = list(label_to_idx.keys())
weights = torch.tensor([1.0] + [10.0] * 14).cuda()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

# Read data from CoNLL format
def read_conll(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    # Append all 0th elements in lines into a list
    sentences = [line[0] for line in lines[1:-1] if len(line) > 0]
    labels = [line[3] for line in lines[1:-1] if len(line) > 0]
    # Break sentences and labels into length 512 chunks
    sentences = [sentences[i:i + 512] for i in range(0, len(sentences), 512)]
    labels = [labels[i:i + 512] for i in range(0, len(labels), 512)]
    return sentences, labels

def calculate_metrics(predictions, labels):
    predicted_labels = [idx_to_label[i] for i in predictions]
    true_labels = [idx_to_label[i] for i in labels]

    # Ignore the "O" tag for metrics
    true_labels = [label for label in true_labels if label != 'O']
    predicted_labels = [predicted for label, predicted in zip(true_labels, predicted_labels) if label != 'O']

    results = seqeval.compute(predictions=[predicted_labels], references=[true_labels])
    print(results)

def tokenize_and_convert(sentences, labels):
    numerical_sentences = []
    numerical_labels = []
    for sentence, label_seq in zip(sentences, labels):
        numerical_sentence = []
        numerical_label = []
        for word, label in zip(sentence, label_seq):
            tokenized_word = tokenizer.tokenize(word)
            subword_ids = tokenizer.convert_tokens_to_ids(tokenized_word)
            numerical_sentence.extend(subword_ids)
            numerical_label.extend([label_to_idx[label]] * len(subword_ids))
        numerical_sentences.append(numerical_sentence)
        numerical_labels.append(numerical_label)
    return numerical_sentences, numerical_labels

def create_data_loader(X, y, batch_size):
    inputs = [torch.LongTensor(x) for x in X]
    labels = [torch.LongTensor(y_seq) for y_seq in y]

    # Pad sequences to the same length in each batch
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    # Convert to packed sequences for efficient training
    packed_inputs = pack_padded_sequence(padded_inputs, torch.tensor([len(x) for x in X], dtype=torch.long), batch_first=True, enforce_sorted=False)
    packed_labels = pack_padded_sequence(padded_labels, torch.tensor([len(y) for y in y], dtype=torch.long), batch_first=True, enforce_sorted=False)
    attention_masks = (packed_inputs.data != 0).float()
    dataset = TensorDataset(packed_inputs.data, attention_masks, packed_labels.data)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class NERLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(NERLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, attention_mask):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        # Apply attention mask to set padding tokens to zero
        output = output * attention_mask.unsqueeze(-1)
        return output

def train_scierc():
    X_train, y_train = read_conll("AnnotatedData/train_scierc.conll")
    X_test, y_test = read_conll("AnnotatedData/test_scierc.conll")
    X_val, y_val = read_conll("AnnotatedData/dev_scierc.conll")
    X_train, y_train = tokenize_and_convert(X_train, y_train)
    X_val, y_val = tokenize_and_convert(X_val, y_val)
    X_test, y_test = tokenize_and_convert(X_test, y_test)

    batch_size = 32
    train_loader = create_data_loader(X_train, y_train, batch_size)
    val_loader = create_data_loader(X_val, y_val, batch_size)
    test_loader = create_data_loader(X_test, y_test, batch_size)

    embedding_dim = 100
    hidden_dim = 256
    output_dim = len(label_to_idx)
    model = NERLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_attention_mask, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x, batch_attention_mask)
            loss = criterion(predictions.view(-1, output_dim).cuda(), batch_y.view(-1).cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
        # print precision, recall and accuracy on validation set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_attention_mask, batch_y in val_loader:
                predictions = model(batch_x, batch_attention_mask)
                all_preds.extend(predictions.argmax(dim=-1).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Calculate and print validation metrics
        calculate_metrics(all_preds, all_labels)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_attention_mask, batch_y in test_loader:
            predictions = model(batch_x, batch_attention_mask)
            all_preds.extend(predictions.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate and print validation metrics
    calculate_metrics(all_preds, all_labels)
    return model

def finetune(model):
    X_train, y_train = read_conll("AnnotatedData/data_temp.conll")
    X_test, y_test = read_conll("AnnotatedData/data_temp.conll")
    X_val, y_val = read_conll("AnnotatedData/data_temp.conll")
    X_train, y_train = tokenize_and_convert(X_train, y_train)
    X_val, y_val = tokenize_and_convert(X_val, y_val)
    X_test, y_test = tokenize_and_convert(X_test, y_test)

    batch_size = 32
    train_loader = create_data_loader(X_train, y_train, batch_size)
    val_loader = create_data_loader(X_train, y_train, batch_size)
    test_loader = create_data_loader(X_test, y_test, batch_size)

    output_dim = len(label_to_idx)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_attention_mask, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x, batch_attention_mask)
            loss = criterion(predictions.view(-1, output_dim).cuda(), batch_y.view(-1).cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
        # print precision, recall and accuracy on validation set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_attention_mask, batch_y in val_loader:
                predictions = model(batch_x, batch_attention_mask)
                all_preds.extend(predictions.argmax(dim=-1).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Calculate and print validation metrics
        calculate_metrics(all_preds, all_labels)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_attention_mask, batch_y in test_loader:
            predictions = model(batch_x, batch_attention_mask)
            all_preds.extend(predictions.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate and print validation metrics
    calculate_metrics(all_preds, all_labels)


def main():
    model = train_scierc()
    finetune(model)


if __name__ == "__main__":
    main()
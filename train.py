# The "AnnotatedData" folder contains NER data in CoNLL format.
# Read from that folder and train a huggingface AutoModelForTokenClassification model.
# Print training metrics and performance of model on each class

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

# Read data from CoNLL format
def read_conll(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    return lines

# Convert data to huggingface format
def convert_to_hf(lines):
    hf_data = []
    for line in lines:
        if len(line) == 0:
            continue
        word, tag = line[0], line[1]
        hf_data.append((word, tag))
    return hf_data

# define a function to tokenize the data
def tokenize_and_align_labels(examples):
    # Define a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# define a function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = classification_report(true_labels, true_predictions, digits=4)
    return results

# define a function to train the model
def train_model(train_data, eval_data, label_list):
    # Load the model
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )
    # Train the model
    trainer.train()
    # Evaluate the model
    trainer.evaluate()
    # Save the model
    trainer.save_model("./models")

# define a function to predict on test data
def predict_on_test(test_data, label_list):
    # Load the model
    model = AutoModelForTokenClassification.from_pretrained("./models")
    # Define the trainer
    trainer = Trainer(model=model)
    # Predict on test data
    predictions, labels, metrics = trainer.predict(test_data)
    # Compute classification report
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = classification_report(true_labels, true_predictions, digits=4)
    return results

# define a function to print the results
def print_results(results):
    print("Results on test data:")
    print(results)
    print("\n")
    print("Results on each class:")
    print("Class\tPrecision\tRecall\tF1-score")
    for line in results.split("\n")[2:-5]:
        line = line.split()
        if len(line) == 0:
            continue
        print(line[0], "\t", line[1], "\t", line[2], "\t", line[3])

# Call previous methods
if __name__ == "__main__":

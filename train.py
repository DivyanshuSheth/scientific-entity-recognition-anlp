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
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import evaluate

seqeval = evaluate.load("seqeval")

label_to_id = {"O": 0,
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
id_to_label = {v: k for k, v in label_to_id.items()}
label_list = list(label_to_id.keys())
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Read data from CoNLL format
def read_conll(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    lines = [[line[0], line[3]] for line in lines[1:-1] if len(line) > 0]
    return lines

# Convert data to huggingface format
def convert_to_hf(lines):
    hf_data = []
    for line in lines:
        if len(line) == 0:
            continue
        word, tag = line[0], line[1]
        hf_data.append((word, tag))
    # Combine every 512 elements of hf_data words and tags into a single list
    hf_data = [hf_data[i:i + 512] for i in range(0, len(hf_data), 512)]
    hf_data = [{'id': idx, 'tokens': [x[0] for x in batch], 'ner_tags': [label_to_id[x[1]] for x in batch]} for idx, batch in enumerate(hf_data)]
    return hf_data

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# define a function to train the model
def train_model(ds):
    # Load the model
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(id_to_label), id2label=id_to_label, label2id=label_to_id)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-2,
        num_train_epochs=30,
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
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # Train the model
    trainer.train()
    # Evaluate the model
    trainer.evaluate()
    # Save the model
    trainer.save_model("./models")


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
    # Read data
    train_lines = read_conll("./AnnotatedData/data_temp.conll")
    dev_lines = read_conll("./AnnotatedData/data_temp.conll")
    test_lines = read_conll("./AnnotatedData/data_temp.conll")
    # Convert data to huggingface format
    train_data = convert_to_hf(train_lines)
    dev_data = convert_to_hf(dev_lines)
    test_data = convert_to_hf(test_lines)
    # print(train_data[0])
    # # Convert data to pandas dataframe
    train_df = pd.DataFrame(train_data, columns=["id", "tokens", "ner_tags"])
    dev_df = pd.DataFrame(dev_data, columns=["id", "tokens", "ner_tags"])
    test_df = pd.DataFrame(test_data, columns=["id", "tokens", "ner_tags"])
    trainds = Dataset.from_pandas(train_df)
    testds = Dataset.from_pandas(test_df)
    valds = Dataset.from_pandas(dev_df)

    ds = DatasetDict()

    ds['train'] = trainds
    ds['validation'] = valds
    ds['test'] = testds

    # datasets = load_dataset("conll2003")
    ds = ds.map(tokenize_and_align_labels, batched=True)
    train_model(ds)



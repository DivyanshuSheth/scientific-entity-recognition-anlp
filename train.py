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
from torch.nn import CrossEntropyLoss
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import evaluate

from utility import compute_metrics, read_conll, convert_to_hf, train_val_split, label_to_id, id_to_label, predict, flatten_list, predict_on_file


weights = torch.tensor([1.0] + [10.0] * 14).cuda()
tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

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

class WeightedCrossEntropyTrainer(Trainer):
    def __init__(self, *args, weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.weight).cuda()
        loss = loss_fct(logits.view(-1, model.config.num_labels).cuda(), labels.view(-1).cuda())
        return (loss, outputs) if return_outputs else loss

# define a function to train the model
def train_model(ds, model=None):
    # Load the model
    if model is None:
        model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=len(id_to_label), id2label=id_to_label, label2id=label_to_id)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="/home/scratch/vdas/results_anlp",
        learning_rate=0.0001,
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
    trainer = WeightedCrossEntropyTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        weight=weights
    )
    # Train the model
    trainer.train()
    # Evaluate the model
    trainer.evaluate()
    # Save the model
    # trainer.save_model("./models")
    return model

# Call previous methods
if __name__ == "__main__":
    # Read data
    # lines_scierc = read_conll("./AnnotatedData/train_scierc.conll")
    # lines_scierc_val = read_conll("./AnnotatedData/dev_scierc.conll")
    # lines_scierc = flatten_list(lines_scierc)
    # lines_scierc_val = flatten_list(lines_scierc_val)
    # train_scierc = convert_to_hf(lines_scierc)
    # val_scierc = convert_to_hf(lines_scierc_val)
    # train_scierc_df = pd.DataFrame(train_scierc, columns=["id", "tokens", "ner_tags"])
    # val_scierc_df = pd.DataFrame(val_scierc, columns=["id", "tokens", "ner_tags"])
    # train_scierc_ds = Dataset.from_pandas(train_scierc_df)
    # val_scierc_ds = Dataset.from_pandas(val_scierc_df)
    # ds_scierc = DatasetDict()
    # ds_scierc['train'] = train_scierc_ds
    # ds_scierc['validation'] = val_scierc_ds
    # ds_scierc = ds_scierc.map(tokenize_and_align_labels, batched=True)
    # model = train_model(ds_scierc)
    #
    # lines = read_conll("./AnnotatedData/data.conll")
    # train_lines, dev_lines = train_val_split(lines)
    # # Convert data to huggingface format
    # train_data = convert_to_hf(train_lines)
    # dev_data = convert_to_hf(dev_lines)
    # # print(train_data[0])
    # # # Convert data to pandas dataframe
    # train_df = pd.DataFrame(train_data, columns=["id", "tokens", "ner_tags"])
    # dev_df = pd.DataFrame(dev_data, columns=["id", "tokens", "ner_tags"])
    # trainds = Dataset.from_pandas(train_df)
    # valds = Dataset.from_pandas(dev_df)
    #
    # ds = DatasetDict()
    #
    # ds['train'] = trainds
    # ds['validation'] = valds
    # ds = ds.map(tokenize_and_align_labels, batched=True)
    # model = train_model(ds, model)
    model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=len(id_to_label),
                                                            id2label=id_to_label, label2id=label_to_id)
    predict_on_file("./AnnotatedData/test.csv", model, tokenizer)
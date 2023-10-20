import numpy as np
import evaluate
import itertools

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


# Read data from CoNLL format
def read_conll(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
 
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    lines = [('' if len(line) == 0 else [line[0], line[3]]) for line in lines[1:-1]]

    groups = itertools.groupby(lines, key=lambda x: x != "")
    groups = (list(group) for k, group in groups if k)

    all_papers = [g for g in groups]

    return all_papers

# Convert data to huggingface format
def convert_to_hf(lines):
    hf_data = []
    for line in lines:
        if len(line) == 0:
            continue
        word, tag = line[0], line[1]
        hf_data.append((word, tag))

    # Combine every 512 elements of hf_data words and tags into a single list
    hf_data = [hf_data[i:i + 32] for i in range(0, len(hf_data), 32)]
    hf_data = [{'id': idx, 'tokens': [x[0] for x in batch], 'ner_tags': [label_to_id[x[1]] for x in batch]} for idx, batch in enumerate(hf_data)]
    return hf_data

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

    true_labels = [label for label in true_labels if label != 'O']
    true_predictions = [predicted for label, predicted in zip(true_labels, true_predictions) if label != 'O']
    print(true_labels, true_predictions)
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train_val_split(list_of_paper_lines):
    val_indices = np.random.choice(len(list_of_paper_lines), size=5, replace=False)
    train_indices = [i for i in range(len(list_of_paper_lines)) if i not in val_indices]

    train_lines = [list_of_paper_lines[i] for i in train_indices]
    val_lines = [list_of_paper_lines[i] for i in val_indices]

    train_lines = [line for paper in train_lines for line in paper]
    val_lines = [line for paper in val_lines for line in paper]

    return train_lines, val_lines
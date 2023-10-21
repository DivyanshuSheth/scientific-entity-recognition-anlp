import numpy as np
import pandas as pd
import evaluate
import itertools
import torch

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
    # Divide list by the blank strings
    lines = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '') if not x]
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
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print_results(results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def print_results(results):
    # print("Results on test data:")
    # print(results)
    # print("\n")
    # print("Results on each class:")
    # print("Class\tPrecision\tRecall\tF1-score")
    # for classname, class_results in results.items():
    #     if not isinstance(class_results, dict):
    #         continue
    #     print(classname, class_results)
    #     print(classname, "\t", class_results["precision"], "\t", class_results["recall"], "\t", class_results["f1"])

    columns = list(results.values())[0].keys()
    column_widths = {col: len(col) for col in columns}

    for row in results.values():
        if not isinstance(row, dict):
            continue
        for col, value in row.items():
            column_widths[col] = max(column_widths[col], len(str(value)))

    # Print the table
    header = "| ".join(["Row Name"] + [col.ljust(column_widths[col]) for col in columns])
    print(header)

    for row_name, row_data in results.items():
        formatted_row = "| ".join([row_name if isinstance(row_data, dict) else ''] + [str(row_data[col]).ljust(column_widths[col]) for col in columns if isinstance(row_data, dict)])
        if len(formatted_row) > 0:
            print(formatted_row)
    print('Overall precision: ', results['overall_precision'])
    print('Overall recall: ', results['overall_recall'])
    print('Overall f1: ', results['overall_f1'])
    print('Overall accuracy: ', results['overall_accuracy'])
    print('\n')


def train_val_split(list_of_paper_lines):
    val_indices = np.random.choice(len(list_of_paper_lines), size=5, replace=False)
    train_indices = [i for i in range(len(list_of_paper_lines)) if i not in val_indices]

    train_lines = [list_of_paper_lines[i] for i in train_indices]
    val_lines = [list_of_paper_lines[i] for i in val_indices]

    train_lines = [line for paper in train_lines for line in paper]
    val_lines = [line for paper in val_lines for line in paper]

    return train_lines, val_lines

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def predict(model, tokenizer, sentence):
    tokenized_sentence = tokenizer.encode(sentence)
    input_ids = torch.tensor([tokenized_sentence]).cuda()
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for idx, (token, label_idx) in enumerate(zip(tokens, label_indices[0])):
        if idx == 0 or idx == len(tokens) - 1:
            continue
        if not token.startswith("Ġ"):
            new_tokens[-1] = new_tokens[-1] + token
        else:
            new_labels.append(id_to_label[label_idx])
            new_tokens.append(token[1:])
    return [(token, label) for token, label in zip(new_tokens, new_labels)]

def predict_on_file(filepath, model, tokenizer):
    # Read csv
    df = pd.read_csv(filepath)
    pred_ids = []
    pred_labels = []
    sentence = ''
    i = 0
    for index, row in df.iterrows():
        if index == 0:
            continue # ignore docstart
        if pd.isna(row['input']) or index % 32 == 0:
            preds = predict(model.cuda(), tokenizer, sentence[:-1])
            pred_labels.extend([pred[1] for pred in preds])
            assert i == len(preds)
            i = 0
            sentence = ''
            if pd.isna(row['input']):
                pred_ids.append(row['id'])
                pred_labels.append('X')
        else:
            pred_ids.append(row['id'])
            sentence += row['input'] + ' '
            i += 1
    preds = predict(model.cuda(), tokenizer, sentence[:-1])
    pred_labels.extend([pred[1] for pred in preds])
    pred_df = pd.DataFrame({'id': pred_ids, 'target': pred_labels})
    pred_df.to_csv('predictions.csv', index=False)
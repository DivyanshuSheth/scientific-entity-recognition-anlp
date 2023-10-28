from utility import read_conll, train_val_split, flatten_list
import pandas as pd
from collections import defaultdict
import evaluate

def train(train_df):
    #strip
    # train_df['ner_tags'] = train_df['ner_tags'].str.strip("IB-")
    unique_tags = train_df['ner_tags'].unique()
    
    #count unique tags
    unique_tags_count = train_df['ner_tags'].value_counts()
    print(unique_tags_count)

    seen_tokens = defaultdict(dict)
    for tag in unique_tags:
        unique_tokens_count = train_df[train_df['ner_tags'] == tag]['tokens'].value_counts()
        
        #divide by total
        unique_tokens_count = unique_tokens_count / unique_tags_count[tag]
        
        seen_tokens[tag] = {token: count for token, count in zip(unique_tokens_count.index, unique_tokens_count.values)}

    return seen_tokens

def eval(seen_tokens, val_df, out_path):
    for tag in seen_tokens.keys():
        val_df[tag] = val_df['tokens'].apply(lambda x: seen_tokens[tag].get(x, 0))

    pred = val_df[seen_tokens.keys()].idxmax(axis=1)
    val_df['pred'] = pred

    #remove I if not connected to a relevant 'B'
    idx = 0
    while idx < len(val_df):
        prefix = val_df.iloc[idx]['pred'][0]
        suffix = val_df.iloc[idx]['pred'][1:]
        
        if prefix == 'B':
            idx += 1
            next_prefix = val_df.iloc[idx]['pred'][0]
            next_suffix = val_df.iloc[idx]['pred'][1:] 
            while idx < len(val_df) and next_prefix == 'I' and next_suffix == suffix:
                idx += 1
                next_prefix = val_df.iloc[idx]['pred'][0]
                next_suffix = val_df.iloc[idx]['pred'][1:]
        elif prefix == 'I':
            val_df.iloc[idx, val_df.columns.get_loc('pred')] = 'O'
            idx += 1
        else:
            idx += 1

    correct = val_df['pred'] == val_df['ner_tags']
    print(correct.value_counts())

    correct_wo_O = correct[val_df['ner_tags'] != 'O']
    print(correct_wo_O.value_counts())

    seqeval = evaluate.load("seqeval")
    true_labels = val_df['ner_tags'].tolist()
    pred_labels = val_df['pred'].tolist()

    true_labels_witho = [[label] for label in true_labels]
    pred_labels_witho = [[pred] for pred in pred_labels]

    true_labels_withouto = [[label] for label in true_labels if label != 'O']
    pred_labels_withouto  = [[pred] for (pred, label) in zip(pred_labels, true_labels) if label != 'O']

    results_witho = seqeval.compute(predictions=pred_labels_witho, references=true_labels_witho)
    res_witho = {"precision": results_witho["overall_precision"],
        "recall": results_witho["overall_recall"],
        "f1": results_witho["overall_f1"],
        "accuracy": results_witho["overall_accuracy"],}
    
    results_withouto = seqeval.compute(predictions=pred_labels_withouto, references=true_labels_withouto)
    res_withouto = {"precision": results_withouto["overall_precision"],
        "recall": results_withouto["overall_recall"],
        "f1": results_withouto["overall_f1"],
        "accuracy": results_withouto["overall_accuracy"],}
    
    print(f"Results with O: {res_witho}")
    print(f"Results without O: {res_withouto}")

    #save results
    val_df_final = val_df[['tokens', 'ner_tags', 'pred']]
    val_df_final.to_csv(out_path)


if __name__ == "__main__":
    train_lines = read_conll("./FinalData/train.conll")
    train_lines = flatten_list(train_lines)
    val_lines = read_conll("./FinalData/val.conll")
    val_lines = flatten_list(val_lines)

    train_df= pd.DataFrame(train_lines, columns=["tokens", "ner_tags"])
    val_df = pd.DataFrame(val_lines, columns=["tokens", "ner_tags"])

    seen_tokens = train(train_df)
    eval(seen_tokens, val_df, out_path="ValidationResults/baseline_val_results.csv")
    eval(seen_tokens, train_df, out_path="ValidationResults/baseline_train_results.csv")

import pandas as pd
import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")

def load_transformer_results(results_path, label_path):
    results_df = pd.read_csv(results_path)
    label_df = pd.read_csv(label_path)
    results_df.set_index('id', inplace=True)
    label_df.set_index('id', inplace=True)

    results_df = results_df.merge(label_df, on='id', how='left')
    results_df = results_df[results_df['target_x'] != 'X']

    results_df.rename(columns={'target_x': 'pred', 'target_y': 'ner_tags'}, inplace=True)

    return results_df

def performance(df):
    classes = df['ner_tags'].unique()
    classes = [c for c in classes if c != 'O']
    classes.sort()

    true_labels = df['ner_tags'].tolist()
    pred_labels = df['pred'].tolist()

    true_labels = [[label] for label in true_labels]
    pred_labels = [[pred] for pred in pred_labels]

    results = seqeval.compute(predictions=pred_labels, references=true_labels)

    return results

def format_latex_table(list_of_results, method_names, metric):
    #performance by class
    latex_table = "\\begin{table}[] \n"

    latex_table += "    \\begin{tabular}{|l|c|c|c|}\n"
    latex_table += "        \\hline \n"

    class_names = list(list_of_results[0].keys())
    class_names = [c for c in class_names if 'overall' not in c]
    class_names.sort()
    latex_table += '        &'
    latex_table += ' & '.join(method_names)
    latex_table += "\\\\ \\hline \n"


    for class_name in class_names:
        latex_table += f"        {class_name}"

        best_val = 0
        best_method = None
        for results, method_name in zip(list_of_results, method_names):
            val = results[class_name][metric]
            if val > best_val:
                best_val = val
                best_method = method_name

        for results, method_name in zip(list_of_results, method_names):
            val = results[class_name][metric]
            if method_name == best_method:
                latex_table += f" & \\textbf{{{val: 0.3f}}}"
            else:
                latex_table += f" & {val: 0.3f}"
        
        latex_table += "\\\\\n        \\hline \n"
    
    latex_table += "    \\end{tabular}\n"

    latex_table += f"    \\caption{{  {metric} by class for each method. \\hfill\\quad }}\n"
    latex_table += f"    \\label{{table:{metric}_by_class}}\n"
    latex_table += "\\end{table}\n"

    print(latex_table)

import matplotlib.pyplot as plt
def confusion_matrix(df, name):
    #remove B and I
    df['ner_tags'] = df['ner_tags'].apply(lambda x: x[2:] if x.startswith('B-') or x.startswith('I-') else x)
    df['pred'] = df['pred'].apply(lambda x: x[2:] if x.startswith('B-') or x.startswith('I-') else x)

    classes = df['ner_tags'].unique()
    
    classes.sort()
    classes = [c for c in classes if c != 'O']
    classes.append('O')
    
    N = len(classes)

    true_labels = df['ner_tags'].tolist()
    pred_labels = df['pred'].tolist()

    #plot confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    #separate cm for o, since it's so much larger

    fig, ax = plt.subplots(figsize=(9, 6))
    #red to blue
    # im = ax.imshow(cm, cmap=plt.cm.magma)
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]    

    im = ax.imshow(cm_norm, cmap=plt.cm.magma)

    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(N):
        for j in range(N):
            ax.text(j, i, int(cm[i, j]),ha="center", va="center", color="w" if cm_norm[i, j] < np.max(cm_norm) / 3 else "k")

    ax.set_title(f"{name}", fontsize=15, fontweight='bold')

    fig.tight_layout()

    name_sanitized = name.replace(' ', '_').replace('-', '').replace('/', '').lower()
    fig.savefig(f"Plots/{name_sanitized}_confusion_matrix.png")




if __name__ == "__main__":
    train_label_path = "FinalData/train.csv"
    val_label_path = "FinalData/val.csv"

    train_baseline_results_path = "ValidationResults/baseline_train_results.csv"
    train_baseline_results_df = pd.read_csv(train_baseline_results_path)
    val_baseline_results_path = "ValidationResults/baseline_val_results.csv"
    val_baseline_results_df = pd.read_csv(val_baseline_results_path)

    train_roberta_pretrain_path = "ValidationResults/train_predictions_ours_roberta_pretrain.csv"
    roberta_pretrain_train_results_df = load_transformer_results(train_roberta_pretrain_path, train_label_path)
    val_roberta_pretrain_path = "ValidationResults/val_predictions_ours_roberta_pretrain.csv"
    roberta_pretrain_val_results_df = load_transformer_results(val_roberta_pretrain_path, val_label_path)

    train_roberta_nopretrain_path = "ValidationResults/train_predictions_ours_roberta_nopretrain.csv"
    roberta_nopretrain_train_results_df = load_transformer_results(train_roberta_nopretrain_path, train_label_path)
    val_roberta_nopretrain_path = "ValidationResults/val_predictions_ours_roberta_nopretrain.csv"
    roberta_nopretrain_val_results_df = load_transformer_results(val_roberta_nopretrain_path, val_label_path)


    val_baseline_results = performance(val_baseline_results_df)
    roberta_pretrain_val_results = performance(roberta_pretrain_val_results_df)
    roberta_nopretrain_val_results = performance(roberta_nopretrain_val_results_df)

    method_names = ["Baseline", "RB-large w/o pretraining", "RB-large w/ pretraining"]
    list_of_results = [val_baseline_results, roberta_nopretrain_val_results, roberta_pretrain_val_results]
    # format_latex_table(list_of_results, method_names, metric='f1')

    confusion_matrix(val_baseline_results_df, name="Baseline")
    confusion_matrix(roberta_pretrain_val_results_df, name="RB-large w/ pretraining")
    confusion_matrix(roberta_nopretrain_val_results_df, name="RB-large w/o pretraining")
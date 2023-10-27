
import pandas as pd
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import glob

seqeval = evaluate.load("seqeval")

def get_metrics(bootstrap_sample):
    true_labels = np.array(bootstrap_sample['ner_tags'].tolist())
    pred_labels = np.array(bootstrap_sample['pred'].tolist())

    true_labels_witho = true_labels[None,:]
    pred_labels_witho = pred_labels[None,:]

    results_witho = seqeval.compute(predictions=pred_labels_witho, references=true_labels_witho)
    res_witho = {"precision": results_witho["overall_precision"],
        "recall": results_witho["overall_recall"],
        "f1": results_witho["overall_f1"],
        "accuracy": results_witho["overall_accuracy"],}

    
    return res_witho

def bootstrap(results_df, out_path, N=100):
    bootstrap_res = defaultdict(list)
    
    for i in range(N):
        print(f"Bootstrap iteration {i}")
        bootstrap_sample = results_df.sample(frac=1, replace=True)
        res_witho = get_metrics(bootstrap_sample)
        for k, v in res_witho.items():
            bootstrap_res[k].append(v)

    with open(out_path, 'w') as f:
        json.dump(bootstrap_res, f)


def plot_ci(several_bootstrap_results, experiment_names, alpha=0.05, metric='f1'):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(10, 4))

    for bootstrap_results, experiment_name in zip(several_bootstrap_results, experiment_names):
        scores = bootstrap_results[metric]
        N = len(scores)
        
        scores = sorted(scores)
        lower = scores[int((alpha/2) * N)]
        upper = scores[int((1-alpha/2) * N)]

        color = next(ax._get_lines.prop_cycler)['color']

        ax.hist(scores, bins=20, label=experiment_name, color=color, alpha=0.5)

        ax.axvline(lower, color=color, linestyle='dashed')
        ax.axvline(upper, color=color, linestyle='dashed')

    ax.set_title(f"Distribution of {metric} with 95% CI Using Bootstrapping (n=100)", fontsize=15)
    ax.set_xlabel(f"{metric}", fontsize=15, fontweight='bold')
    ax.set_ylabel("Frequency")
    ax.legend()
    
    fig.savefig(f"Plots/{metric}_bootstrap.png", dpi=300, bbox_inches='tight')

def load_transformer_results(results_path, label_path):
    results_df = pd.read_csv(results_path)
    label_df = pd.read_csv(label_path)
    results_df.set_index('id', inplace=True)
    label_df.set_index('id', inplace=True)

    results_df = results_df.merge(label_df, on='id', how='left')
    results_df = results_df[results_df['target_x'] != 'X']

    results_df.rename(columns={'target_x': 'pred', 'target_y': 'ner_tags'}, inplace=True)

    return results_df


if __name__ == "__main__":
    train_label_path = "FinalData/train.csv"
    val_label_path = "FinalData/val.csv"

    train_baseline_results_path = "ValidationResults/baseline_train_results.csv"
    train_baseline_results_df = pd.read_csv(train_baseline_results_path)
    val_baseline_results_path = "ValidationResults/baseline_val_results.csv"
    val_baseline_results_df = pd.read_csv(val_baseline_results_path)

    train_roberta_finetune_path = "ValidationResults/train_predictions_ours_roberta_finetune.csv"
    roberta_finetune_train_results_df = load_transformer_results(train_roberta_finetune_path, train_label_path)
    val_roberta_finetune_path = "ValidationResults/val_predictions_ours_roberta_finetune.csv"
    roberta_finetune_val_results_df = load_transformer_results(val_roberta_finetune_path, val_label_path)

    train_roberta_nofinetune_path = "ValidationResults/train_predictions_ours_roberta_nofinetune.csv"
    roberta_nofinetune_train_results_df = load_transformer_results(train_roberta_nofinetune_path, train_label_path)
    val_roberta_nofinetune_path = "ValidationResults/val_predictions_ours_roberta_nofinetune.csv"
    roberta_nofinetune_val_results_df = load_transformer_results(val_roberta_nofinetune_path, val_label_path)

    # bootstrap(train_baseline_results_df, out_path="BootstrapResults/baseline__train_bootstrap.json")
    # bootstrap(val_baseline_results_df, out_path="BootstrapResults/baseline__val_bootstrap.json")
    # bootstrap(roberta_finetune_train_results_df, out_path="BootstrapResults/roberta_pretrain_train_bootstrap.json")
    # bootstrap(roberta_finetune_val_results_df, out_path="BootstrapResults/roberta_pretrain_val_bootstrap.json")
    # bootstrap(roberta_nofinetune_train_results_df, out_path="BootstrapResults/roberta_nopretrain_train_bootstrap.json")
    # bootstrap(roberta_nofinetune_val_results_df, out_path="BootstrapResults/roberta_nopretrain_val_bootstrap.json")

    several_bootstrap_results_paths = sorted(glob.glob("BootstrapResults/*.json"))
    several_bootstrap_results = []
    experiment_names = []
    for path in several_bootstrap_results_paths:
        filename = path.split('/')[-1].split('.')[0].split('_')

        if filename[2] == 'train':
            continue

        experiment_name = f"{filename[0]} {filename[1]} - {filename[2]}"
        experiment_names.append(experiment_name)

        
        with open(path, 'r') as f:
            bootstrap_results = json.load(f)
            several_bootstrap_results.append(bootstrap_results)
        
    plot_ci(several_bootstrap_results, experiment_names, metric='f1')
    plot_ci(several_bootstrap_results, experiment_names, metric='accuracy')
    plot_ci(several_bootstrap_results, experiment_names, metric='recall')
    plot_ci(several_bootstrap_results, experiment_names, metric='precision')
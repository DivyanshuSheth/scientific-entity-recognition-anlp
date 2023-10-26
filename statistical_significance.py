
import pandas as pd
import evaluate
import numpy as np
import matplotlib.pyplot as plt

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

def bootstrap_ci(results_df, N=100, alpha=0.05):
    f1_scores = []
    
    for i in range(N):
        print(f"Bootstrap iteration {i}")
        bootstrap_sample = results_df.sample(frac=1, replace=True)
        res_witho = get_metrics(bootstrap_sample)
        f1_scores.append(res_witho['f1'])

    
    f1_scores = sorted(f1_scores)
    lower = f1_scores[int((alpha/2) * N)]
    upper = f1_scores[int((1-alpha/2) * N)]
    print(f"95% CI: [{lower}, {upper}]")

    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()

    ax.hist(f1_scores, bins=20)

    ax.axvline(lower, color='blue', linestyle='dashed')
    ax.axvline(upper, color='blue', linestyle='dashed')

    ax.set_title("Distribution of F1 scores")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.show()

if __name__ == "__main__":
    baseline_results_path = "ValidationResults/baseline_results.csv"
    baseline_results_df = pd.read_csv(baseline_results_path)

    bootstrap_ci(baseline_results_df)


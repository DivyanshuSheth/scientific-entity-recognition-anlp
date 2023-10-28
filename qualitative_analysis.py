import pandas as pd
import numpy as np

def load_transformer_results(results_path, label_path):
    results_df = pd.read_csv(results_path)
    label_df = pd.read_csv(label_path)
    results_df.set_index('id', inplace=True)
    label_df.set_index('id', inplace=True)

    results_df = results_df.merge(label_df, on='id', how='left')
    results_df = results_df[results_df['target_x'] != 'X']

    results_df.rename(columns={'target_x': 'pred', 'target_y': 'ner_tags'}, inplace=True)

    return results_df

def sample_outputs(baseline_df, pretrain_df, nopretrain_df, out_path):
    #find location of baseline misclassification
    baseline_misclassified = baseline_df[baseline_df['ner_tags'] != baseline_df['pred']]
    #get index of misclassification
    baseline_misclassified = baseline_misclassified.index.tolist()
    #get N random samples
    baseline_misclassified = np.random.choice(baseline_misclassified, 1, replace=False)

    #print 10 surrounding tokens in all three dataframes
    context_idxs = np.arange(baseline_misclassified[0]-10, baseline_misclassified[0]+10)
    baseline_sample = baseline_df.loc[context_idxs]

    print(baseline_sample)

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

    sample_outputs(train_baseline_results_df, roberta_pretrain_train_results_df, roberta_nopretrain_train_results_df, out_path="BootstrapResults/baseline__train_bootstrap.json")
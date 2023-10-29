import pandas as pd
import numpy as np

def load_transformer_results(results_path, label_path):
    results_df = pd.read_csv(results_path)
    label_df = pd.read_csv(label_path)
    results_df.set_index('id', inplace=True)
    label_df.set_index('id', inplace=True)

    results_df = results_df.merge(label_df, on='id', how='left')
    # results_df = results_df[results_df['target_x'] != 'X']

    results_df.rename(columns={'target_x': 'pred', 'target_y': 'ner_tags'}, inplace=True)

    return results_df

def sample_outputs(df_baseline, df_pretrain, df_nopretrain):
    start_idx = 30049
    end_idx = 30055
    #pred
    surrounding_indices = np.arange(start_idx, end_idx)
    baseline_pred = df_baseline.iloc[surrounding_indices]['pred'].tolist()
    pretrain_pred = df_pretrain.iloc[surrounding_indices]['pred'].tolist()
    nopretrain_pred = df_nopretrain.iloc[surrounding_indices]['pred'].tolist()

    baseline_ner_tags = df_baseline.iloc[surrounding_indices]['ner_tags'].tolist()
    tokens = df_baseline.iloc[surrounding_indices]['tokens'].tolist()

    #correct if chars after I and B are the same

    baseline_correct = [pred[1:] == ner_tag[1:] for pred, ner_tag in zip(baseline_pred, baseline_ner_tags)]
    pretrain_correct = [pred[1:] == ner_tag[1:] for pred, ner_tag in zip(pretrain_pred, baseline_ner_tags)]
    nopretrain_correct = [pred[1:] == ner_tag[1:] for pred, ner_tag in zip(nopretrain_pred, baseline_ner_tags)]

    baseline_pred = [f"\\textcolor{{red}}{{{pred}}}" if not correct else pred for pred, correct in zip(baseline_pred, baseline_correct)]
    pretrain_pred = [f"\\textcolor{{red}}{{{pred}}}" if not correct else pred for pred, correct in zip(pretrain_pred, pretrain_correct)]
    nopretrain_pred = [f"\\textcolor{{red}}{{{pred}}}" if not correct else pred for pred, correct in zip(nopretrain_pred, nopretrain_correct)]

    #Format as latex table
    latex_table =f"    \\begin{{tabular}}{{|l||{'c|'*(end_idx - start_idx)}}}\n"
    latex_table += "        \\hline \n"
    latex_table +=f"        Tokens &{'&'.join(tokens)} \\\\"
    latex_table += "        \\hline \n"
    latex_table += "        \\hline \n"
    latex_table +=f"        Ground Truth &{'&'.join(baseline_ner_tags)} \\\\"
    latex_table += "        \\hline \n"
    latex_table += "        \\hline \n"
    latex_table +=f"        Baseline &{'&'.join(baseline_pred)} \\\\"
    latex_table += "        \\hline \n"
    latex_table +=f"        No Pretrain &{'&'.join(nopretrain_pred)} \\\\"
    latex_table += "        \\hline \n"
    latex_table +=f"        Pretrain &{'&'.join(pretrain_pred)} \\\\"
    latex_table += "        \\hline \n"
    latex_table += "    \\end{tabular} \n"

    print(latex_table)

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


    #save val df as csv
    #reorder columns

    roberta_pretrain_val_results_df = roberta_pretrain_val_results_df[['input', 'ner_tags', 'pred']]
    roberta_pretrain_val_results_df.to_csv("ValidationResults/pretrain_combined.csv")

    roberta_nopretrain_val_results_df = roberta_nopretrain_val_results_df[['input', 'ner_tags', 'pred']]
    roberta_nopretrain_val_results_df.to_csv("ValidationResults/nopretrain_combined.csv")

    sample_outputs(val_baseline_results_df, roberta_pretrain_val_results_df, roberta_nopretrain_val_results_df)
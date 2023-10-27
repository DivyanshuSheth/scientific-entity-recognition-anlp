import pandas as pd
from utility import read_conll, flatten_list

train_lines = read_conll("./FinalData/train.conll")
train_lines = flatten_list(train_lines)
val_lines = read_conll("./FinalData/val.conll")
val_lines = flatten_list(val_lines)

train_df= pd.DataFrame(train_lines, columns=["tokens", "ner_tags"])
val_df = pd.DataFrame(val_lines, columns=["tokens", "ner_tags"])

#rename columns to id, input, target
train_df = train_df.rename(columns={"tokens": "input", "ner_tags": "target"})
val_df = val_df.rename(columns={"tokens": "input", "ner_tags": "target"})   

#Use regex to find all input data that looks like "2022.acl-long.1.txt" - dddd.*.*.txt
#Insert "-DOCSTART-" before each of these lines
import re
train_df['input'] = train_df['input'].apply(lambda x: re.sub(r'\d{4}\..*\.txt', '-DOCSTART-', x))
val_df['input'] = val_df['input'].apply(lambda x: re.sub(r'\d{4}\..*\.txt', '-DOCSTART-', x))

#name id column
train_df['id'] = train_df.index
val_df['id'] = val_df.index

#index by id
train_df = train_df.set_index('id')
val_df = val_df.set_index('id')

# #deleta all values from target
# train_df['target'] = train_df['target'].apply(lambda x: '')
# val_df['target'] = val_df['target'].apply(lambda x: '')

train_df.to_csv("./FinalData/train.csv")
val_df.to_csv("./FinalData/val.csv")


# Scientific Named Entity Recognition
## Adiel Felsen, Divyanshu Sheth, Vikramjeet Das

In this project, we build an end-to-end system to perform named entity recognition on scientific texts.

# Data
The train and test sets we use are located at `FinalData/train.conll` and `FinalData/test.conll` respectively.

# Annotation Contributions
Each of us annotated 8 research papers.

Adiel Felsen: 
1. 2022.acl-long.106.txt
2. 2022.emnlp-main.100.txt
3. 2022.emnlp-main.105.txt
4. 2022.emnlp-main.106.txt
5. 2022.emnlp-main.11.txt
6. 2022.naacl-main.100.txt
7. 2023.acl-long.10.txt
8. 2023.acl-long.111.txt

Divyanshu Sheth:
1. 2022.acl-long.480.txt
2. 2022.acl-short.28.txt
3. 2022.acl-short.30.txt
4. 2022.naacl-main.128.txt
5. 2022.naacl-main.59.txt
6. 2023.acl-short.11.txt
7. 2023.acl-short.67.txt
8. 2023.acl-short.69.txt

Vikramjeet Das:
1. 2022.acl-long.1.txt
2. 2022.acl-long.100.txt
3. 2022.acl-long.101.txt
4. 2022.acl-long.102.txt
5. 2022.acl-long.105.txt
6. 2022.acl-long.111.txt
7. 2022.acl-long.141.txt
8. 2022.acl-long.317.txt


# Running the code
Run `python train.py --model_name [model_name] [--pretrain]` to train the model. Replace model_name with the transformer architecture to use (Eg: RoBERTa-large). Add the `finetune` flag to pretrain on SCIERC first.

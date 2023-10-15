import os
import json
import spacy
import glob

from multiprocessing import Pool

def tokenize_sentences(path):
    print(path)
    with open(path, 'r') as f:
        text = f.read()
    nlp = spacy.load('en_core_web_lg')
    tokenizer = nlp.tokenizer
    text = ' '.join([token.text for token in tokenizer(text)])

    text = text.replace('\n', ' ')

    out_path = path.replace('SentenceExtraction', 'Tokenized')  
    with open(out_path, 'w') as f:
        f.write(text)

if __name__ == "__main__":
    paths = glob.glob('SentenceExtraction/*.txt')
    paths = sorted(paths)
    
    pool = Pool()
    pool.map(tokenize_sentences, paths)
    
    
        

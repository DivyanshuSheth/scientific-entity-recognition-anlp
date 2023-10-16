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

def concatenate_into_few_files(base_path):
    paths = glob.glob(base_path + '/*.txt')
    paths = sorted(paths)
    conference_names = [path.split('/')[-1].split('-')[0] for path in paths]
    unique_conference_names = list(set(conference_names))
    print(unique_conference_names)
    
    for conference_name in unique_conference_names:
        paths = glob.glob(os.path.join(base_path, conference_name + '-*.txt'))
        paths = sorted(paths)
        concat = ''
        for path in paths:
            with open(path, 'r') as f:
                text = f.read()
                text = path.split('/')[-1] + text
                
                #delete invalid unicode
                chars = [u'\u0000', u'\u0001', u'\u0002', u'\u0003', u'\u0004', u'\u0005', u'\u0006', u'\u0007', u'\u0008', u'\u000b', u'\u000c', u'\u000e', u'\u000f', u'\u0010', u'\u0011', u'\u0012', u'\u0013', u'\u0014', u'\u0015', u'\u0016', u'\u0017', u'\u0018', u'\u0019', u'\u001a', u'\u001b', u'\u001c', u'\u001d', u'\u001e', u'\u001f']
                for char in chars:
                    text = text.replace(char, '')

                #merge dashed words
                text = text.replace('-   ', '')

                concat += text + '\n'

        out_path = os.path.join(".", conference_name + '-tokenized.txt')

        with open(out_path, 'w') as f:
            f.write(concat)

if __name__ == "__main__":
    paths = glob.glob('SentenceExtraction/*.txt')
    paths = sorted(paths)
    
    # pool = Pool()
    # pool.map(tokenize_sentences, paths)

    concatenate_into_few_files('Tokenized')
    
    
        

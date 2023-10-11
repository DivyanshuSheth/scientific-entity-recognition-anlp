import os
import json
import spacy

def tokenize_sentences(text):
    nlp = spacy.load('en_core_web_lg')
    tokenizer = nlp.tokenizer
    return ' '.join([token.text for token in tokenizer(text)])

if __name__ == "__main__":
    for filename in os.listdir('SentenceExtraction'):
        tokenize_sentences(f'SentenceExtraction/{filename}')
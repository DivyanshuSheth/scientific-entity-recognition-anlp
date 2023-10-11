import os
import json
import spacy


def tokenize_sentences(filename):
    nlp = spacy.load("en_core_web_lg")
    tokenizer = nlp.tokenizer

    t = tokenizer("This is a sentence")
    print(print(len(t)))

    with open(filename, 'r') as f:
        sentences = json.load(f)

    sentence_tokens = []
    for sentence in sentences:
        text = sentence["text"]
        tokens = tokenizer(text)
        print([t for t in tokens])

if __name__ == "__main__":
    for filename in os.listdir('SentenceExtraction'):
        tokenize_sentences(f'SentenceExtraction/{filename}')
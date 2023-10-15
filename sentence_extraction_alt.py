from pypdf import PdfReader
import os
from tokenizer import tokenize_sentences
from tqdm import tqdm

for pdf_path in tqdm(os.listdir('PaperPDF')):
    reader = PdfReader(f"PaperPDF/{pdf_path}")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    sentences = tokenize_sentences(text.replace('\n', ' ').replace('- ', ''))
    with open(f"SentenceExtraction/{pdf_path.replace('.pdf', '.txt')}", 'w') as f:
        f.write(sentences)

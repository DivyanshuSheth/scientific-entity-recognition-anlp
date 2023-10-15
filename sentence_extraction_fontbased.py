import glob
from decimal import Decimal
import PyPDF2

from multiprocessing import Pool

def parse_file(pdf_path):
    print(pdf_path)
    reader = PyPDF2.PdfReader(pdf_path)
    p = 0

    parts = []
    def visitor_body(text, cm, tm, font_dict, font_size):
        #check cm is [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        # print(font_size, text)
        if cm[0] != 1.0 or cm[1] != 0.0 or cm[2] != 0.0 or cm[3] != 1.0:
            return
        
        if Decimal(font_size).compare(Decimal(10.9091)) == 0:
            parts.append(text)
        
        #special exception for abstract
        if Decimal(font_size).compare(Decimal(9.9626)) == 0 and p == 0:
            parts.append(text)
            # parts.append(text)

        # #headers
        if Decimal(font_size).compare(Decimal(11.9552)) == 0:
            parts.append(text)
    for p in range(len(reader.pages)):
        page = reader.pages[p]
        page.extract_text(visitor_text=visitor_body)

    text_body = "".join(parts)

    #save to file
    with open(f"SentenceExtraction/{pdf_path.split('/')[-1][:-4]}.txt", "w") as f:
        f.write(text_body)

if __name__ == "__main__":
    paths = glob.glob("PaperPDF/*.pdf")
    paths = sorted(paths)

    pool = Pool()
    pool.map(parse_file, paths)
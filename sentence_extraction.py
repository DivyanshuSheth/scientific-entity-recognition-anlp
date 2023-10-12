'''
    Extract sentences from PDFs, prepare data to be used by annotater
    Save JSON files to ./SentenceExtraction
'''

import PyPDF2
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
import pdfplumber
import os
import json

from tokenizer import tokenize_sentences


def text_extraction(element):
    # Extracting the text from the in-line text element
    line_text = element.get_text()

    # Find the formats of the text
    # Initialize the list with all the formats that appeared in the line of text
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            # Iterating through each character in the line of text
            for character in text_line:
                if isinstance(character, LTChar):
                    # Append the font name of the character
                    line_formats.append(character.fontname)
                    # Append the font size of the character
                    line_formats.append(character.size)
    # Find the unique font sizes and names in the line
    format_per_line = list(set(line_formats))

    # Return a tuple with the text in each line along with its format
    return (line_text, format_per_line)

for pdf_path in os.listdir('PaperPDF'):
    pdfFileObj = open(f'PaperPDF/{pdf_path}', 'rb')
    pdfRead = PyPDF2.PdfReader(pdfFileObj)

    text_per_page = {}
    for pagenum, page in enumerate(extract_pages(f'PaperPDF/{pdf_path}')):

        pageObj = pdfRead.pages[pagenum]
        page_text = []
        line_format = []
        page_content = []
        first_element = True
        table_extraction_flag = False
        pdf = pdfplumber.open(f'PaperPDF/{pdf_path}')

        page_elements = [(element.y1, element) for element in page._objs]
        page_elements.sort(key=lambda a: a[0], reverse=True)

        for i, component in enumerate(page_elements):
            pos = component[0]
            element = component[1]

            if isinstance(element, LTTextContainer):
                if table_extraction_flag == False:
                    (line_text, format_per_line) = text_extraction(element)
                    page_text.append(line_text)
                    line_format.append(format_per_line)
                    page_content.append(line_text)
                else:
                    # Omit the text that appeared in a table
                    pass

        # Create the key of the dictionary
        dctkey = 'Page_' + str(pagenum)
        # Add the list of list as the value of the page key
        text_per_page[dctkey] = page_content

    # Closing the pdf file object
    pdfFileObj.close()
    paper_text = ''
    for v in text_per_page.values():
        sentences = tokenize_sentences(''.join(v).replace('\n', ' ').replace('- ', ''))
        # print('References' in sentences, sentences)
        if 'References' in sentences:
            break
        paper_text += sentences + ' '
    with open(f'SentenceExtraction/{pdf_path[:-4]}.txt', 'w') as f:
        f.write(paper_text)

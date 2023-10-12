'''
    Scrape ACL Anthology, ArXiv and Semantic Scholar for papers.
    Save PDFs to ./PaperPDF
'''

from bs4 import BeautifulSoup
import requests
import os
import re
import json

# Get all papers from ACL Anthology
def get_papers(acl_anthology_base, conference, year):
    papers = []

    url = acl_anthology_base + os.path.join("events", f"{conference}-{year}")
    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    #all divs with ids {year}{conference}{main or long or short}
    papers_div = soup.find_all("div", id=re.compile(f"{year}{conference}-(main|long|short)"))

    #find all pdf links
    for div in papers_div:
        pdf_links = div.find_all("a", href=re.compile(r'.pdf$'))
        for pdf_link in pdf_links:
            pdf_url = pdf_link["href"]
            pdf_title = pdf_link.text
            papers.append(pdf_url)        

    return papers

if __name__ == "__main__":
    acl_anthology_base = "https://aclanthology.org/"
    acl_2022 = get_papers(acl_anthology_base, "acl", "2022")
    acl_2023 = get_papers(acl_anthology_base, "acl", "2023")
    emnlp_2022 = get_papers(acl_anthology_base, "emnlp", "2022")
    naacl_2022 = get_papers(acl_anthology_base, "naacl", "2022")

    all_paper_urls = {}

    all_paper_urls["acl_2022"] = acl_2022
    all_paper_urls["acl_2023"] = acl_2023
    all_paper_urls["emnlp_2022"] = emnlp_2022
    all_paper_urls["naacl_2022"] = naacl_2022

    #save to json
    with open("all_paper_urls.json", "w") as f:
        json.dump(all_paper_urls, f, indent=4)
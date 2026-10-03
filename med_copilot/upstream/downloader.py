import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://www.nice.org.uk/guidance/"
OUTPUT_DIR = "nice_guidelines"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_nice_guideline(gid):
    url = BASE_URL + gid
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch {gid}")
        return

    soup = BeautifulSoup(r.text, "html.parser")
    # 找 PDF 下载链接
    pdf_link = soup.find("a", {"class": "icon icon--pdf"})
    if pdf_link:
        pdf_url = pdf_link["href"]
        pdf_name = os.path.join(OUTPUT_DIR, f"{gid}.pdf")
        pdf_data = requests.get(pdf_url)
        with open(pdf_name, "wb") as f:
            f.write(pdf_data.content)
        print(f"Downloaded {gid} → {pdf_name}")
    else:
        print(f"No PDF found for {gid}")


# 示例：批量下载前 10 个指南
for i in range(1, 11):
    gid = f"NG{i}"
    download_nice_guideline(gid)

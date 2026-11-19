import os
import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import pdfplumber
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # consistent results

# === CONFIG ===
START_URLS = [
    "https://www.lawethiopia.com",
    "https://chilot.wordpress.com",
    "https://www.fsc.gov.et",
    "https://ecma.gov.et",
    "https://www.agp.gov.et",
    "https://www.fanamc.com",
    "https://borkena.com",
    "https://justice.gov.et",
    "https://www.worldlii.org",
    "https://etelsa.org",
    "https://www.ethiopianembassy.org",
    "https://www.nbe.gov.et",
    ""
]

OUTPUT_DIR = "all_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# === FUNCTIONS ===

def get_domain(url):
    return urlparse(url).netloc

def scrape_pdfs_from_url(base_url, max_pages=10):
    visited = set()
    to_visit = [base_url]
    domain = get_domain(base_url)
    pdf_links = set()

    print(f"\n🔍 Scanning: {base_url}")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=10, headers=HEADERS)
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract all PDF links
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(url, href)
            if full_url.lower().endswith(".pdf"):
                pdf_links.add(full_url)

        # Extract internal links to follow
        for link in soup.find_all("a", href=True):
            href = link["href"]
            next_url = urljoin(url, href)
            if domain in next_url and next_url not in visited and next_url.startswith("http"):
                to_visit.append(next_url)

        time.sleep(1)

    return pdf_links

def download_pdf(url):
    try:
        response = requests.get(url, stream=True, timeout=15, headers=HEADERS)
        if response.status_code == 200:
            filename = os.path.basename(urlparse(url).path)
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filepath
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")
        return None
    return None

def is_amharic_pdf(filepath):
    """Return True if the PDF is in Amharic, or if text extraction fails (to avoid deleting)."""
    try:
        with pdfplumber.open(filepath) as pdf:
            text = ""
            for page in pdf.pages[:3]:  # first 3 pages only
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            if not text.strip():
                # Could be scanned PDF; keep it
                return True
            lang = detect(text)
            return lang == "am"
    except Exception as e:
        print(f"⚠️ Language detection failed for {filepath}: {e}")
        return True  # keep PDF if detection fails

# === MAIN ===

def main():
    all_pdfs = set()
    for site in START_URLS:
        pdfs = scrape_pdfs_from_url(site)
        all_pdfs.update(pdfs)

    print(f"\n📚 Found {len(all_pdfs)} PDF(s). Downloading...")

    for pdf_url in tqdm(all_pdfs):
        path = download_pdf(pdf_url)
        if path and is_amharic_pdf(path):
            tqdm.write(f"✅ Saved PDF: {path}")
        else:
            tqdm.write(f"⚠️ Saved PDF (language unknown or not Amharic): {path or pdf_url}")

    print(f"\n🎉 Done! All PDFs saved in '{OUTPUT_DIR}/' folder.")

if __name__ == "__main__":
    main()

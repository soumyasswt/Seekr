# crawler_indexer.py
import requests
from bs4 import BeautifulSoup
import re
import json
from collections import defaultdict
from urllib.parse import urljoin, urlparse

# -------- CONFIG --------
SEED_URLS = [
    "https://example.com"  # Replace with real starting URLs
]
MAX_PAGES = 50  # Limit to avoid crawling entire internet in prototype
STOPWORDS = {"is", "a", "for", "the", "and", "of", "to", "in"}
INDEX_FILE = "web_index.json"

# -------- HELPERS --------
def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS]

def extract_text(url):
    try:
        res = requests.get(url, timeout=5)
        if "text/html" not in res.headers.get("Content-Type", ""):
            return "", ""
        soup = BeautifulSoup(res.text, "html.parser")
        # Remove scripts/styles
        for s in soup(["script", "style"]):
            s.decompose()
        text = soup.get_text(separator=" ")
        title = soup.title.string if soup.title else url
        return text, title
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return "", ""

# -------- CRAWLER + INDEXER --------
visited = set()
to_visit = list(SEED_URLS)
inverted_index = defaultdict(dict)  # {token: {url: freq}}
page_titles = {}  # store titles for search display

while to_visit and len(visited) < MAX_PAGES:
    url = to_visit.pop(0)
    if url in visited:
        continue
    print(f"Crawling: {url}")
    visited.add(url)

    text, title = extract_text(url)
    page_titles[url] = title
    tokens = tokenize(text)

    # Build index
    for token in tokens:
        if url in inverted_index[token]:
            inverted_index[token][url] += 1
        else:
            inverted_index[token][url] = 1

    # Find new links
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        for link_tag in soup.find_all("a", href=True):
            link = urljoin(url, link_tag['href'])
            # Keep only http/https links
            if urlparse(link).scheme in {"http", "https"} and link not in visited:
                to_visit.append(link)
    except:
        pass

# -------- SAVE INDEX --------
data = {
    "index": inverted_index,
    "titles": page_titles
}
with open(INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Indexed {len(visited)} pages and saved to {INDEX_FILE}")
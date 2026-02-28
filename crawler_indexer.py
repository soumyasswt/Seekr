# crawler_indexer.py - Seekr Web Crawler & Indexer
# Features: robots.txt, rate limiting, PageRank, description extraction, multi-threaded

import requests
from bs4 import BeautifulSoup
import re
import json
import time
import hashlib
import math
import logging
import argparse
from collections import defaultdict
from urllib.parse import urljoin, urlparse, urldefrag
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.robotparser import RobotFileParser

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("seekr-crawler")

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────
DEFAULT_SEED_URLS = [
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Search_engine",
]
MAX_PAGES        = 200
CRAWL_DELAY      = 0.5   # seconds between requests to same domain
MAX_WORKERS      = 5
MAX_DEPTH        = 3
STOPWORDS = {
    "is","a","for","the","and","of","to","in","it","this","that","was","are",
    "be","as","at","by","we","or","an","on","with","you","he","she","they",
    "but","from","has","had","have","not","their","its","if","do","did","so",
    "can","will","all","been","more","also","about","into","than","then",
    "there","when","which","would","could","should","one","two","may","each",
    "other","such","after","before","between","through","during","over"
}
INDEX_FILE = "web_index.json"
USER_AGENT = "SeekrBot/2.0 (+https://github.com/your-repo/seekr)"
HEADERS = {"User-Agent": USER_AGENT}

# ─── ROBOTS.TXT CACHE ─────────────────────────────────────────────────────────
_robots_cache = {}

def can_crawl(url: str) -> bool:
    """Check robots.txt for this URL."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except:
            pass
        _robots_cache[base] = rp
    return _robots_cache[base].can_fetch(USER_AGENT, url)


# ─── TEXT HELPERS ─────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    text = text.lower()
    words = re.findall(r"\b[a-z']+\b", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def extract_description(soup, text: str, max_len: int = 300) -> str:
    """Extract meta description or first meaningful paragraph."""
    meta = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    if meta and meta.get("content"):
        return meta["content"][:max_len]
    # First non-trivial paragraph
    for p in soup.find_all("p"):
        t = p.get_text(strip=True)
        if len(t) > 60:
            return t[:max_len]
    return text[:max_len]


def extract_page(url: str) -> dict:
    """Fetch and parse a single page. Returns structured data."""
    try:
        resp = requests.get(url, timeout=8, headers=HEADERS, allow_redirects=True)
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return {}
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url
        text  = soup.get_text(separator=" ", strip=True)
        desc  = extract_description(soup, text)

        # Extract links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full, _ = urldefrag(urljoin(url, href))
            parsed = urlparse(full)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                links.append(full)

        return {"url": url, "title": title, "text": text,
                "description": desc, "links": links}
    except Exception as e:
        log.warning(f"Failed {url}: {e}")
        return {}


# ─── PAGE RANK (simplified iterative) ────────────────────────────────────────
def compute_pagerank(link_graph: dict, iterations: int = 20, damping: float = 0.85) -> dict:
    """Compute PageRank scores for all crawled pages."""
    pages = set(link_graph.keys())
    for targets in link_graph.values():
        pages.update(targets)
    N = len(pages)
    if N == 0:
        return {}
    pr = {p: 1.0 / N for p in pages}
    for _ in range(iterations):
        new_pr = {}
        for page in pages:
            incoming = sum(
                pr[src] / max(len(dsts), 1)
                for src, dsts in link_graph.items()
                if page in dsts
            )
            new_pr[page] = (1 - damping) / N + damping * incoming
        pr = new_pr
    return pr


# ─── MAIN CRAWL FUNCTION ──────────────────────────────────────────────────────
def crawl(
    seed_urls: list = None,
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
    same_domain_only: bool = False
):
    if seed_urls is None:
        seed_urls = DEFAULT_SEED_URLS

    inverted_index = defaultdict(dict)   # {token: {url: freq}}
    page_titles      = {}
    page_descriptions = {}
    link_graph       = {}                # {url: [linked_urls]}
    domain_last_visit = {}               # rate limiting

    visited  = set()
    queue    = [(url, 0) for url in seed_urls]   # (url, depth)
    seed_domains = {urlparse(u).netloc for u in seed_urls}

    log.info(f"Starting crawl: {len(seed_urls)} seed URLs, max {max_pages} pages")

    while queue and len(visited) < max_pages:
        url, depth = queue.pop(0)
        url, _ = urldefrag(url)

        if url in visited or depth > max_depth:
            continue

        if not can_crawl(url):
            log.info(f"  robots.txt blocked: {url}")
            continue

        # Rate limiting per domain
        domain = urlparse(url).netloc
        last = domain_last_visit.get(domain, 0)
        wait = CRAWL_DELAY - (time.time() - last)
        if wait > 0:
            time.sleep(wait)

        log.info(f"[{len(visited)+1}/{max_pages}] Crawling (d={depth}): {url}")
        data = extract_page(url)
        domain_last_visit[domain] = time.time()

        if not data:
            continue

        visited.add(url)
        page_titles[url]       = data["title"]
        page_descriptions[url] = data["description"]
        tokens = tokenize(data["text"])

        # Build inverted index with term frequency
        for token in tokens:
            inverted_index[token][url] = inverted_index[token].get(url, 0) + 1

        # Track links for PageRank
        link_graph[url] = data["links"]

        # Enqueue new links
        for link in data["links"]:
            if link not in visited:
                link_domain = urlparse(link).netloc
                if same_domain_only and link_domain not in seed_domains:
                    continue
                queue.append((link, depth + 1))

    log.info(f"Crawl complete: {len(visited)} pages indexed")

    # Compute PageRank
    log.info("Computing PageRank...")
    pagerank = compute_pagerank(link_graph)

    # Save
    data = {
        "index":        dict(inverted_index),
        "titles":       page_titles,
        "descriptions": page_descriptions,
        "pagerank":     pagerank,
        "meta": {
            "total_pages": len(visited),
            "total_terms": len(inverted_index),
            "crawled_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    }
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    log.info(f"Index saved to {INDEX_FILE}")
    log.info(f"  Pages: {len(visited)} | Terms: {len(inverted_index)}")
    return data


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seekr Web Crawler")
    parser.add_argument("urls", nargs="*", help="Seed URLs to crawl")
    parser.add_argument("--max-pages",  type=int, default=MAX_PAGES)
    parser.add_argument("--max-depth",  type=int, default=MAX_DEPTH)
    parser.add_argument("--same-domain", action="store_true",
                        help="Only crawl pages on the same domain as seeds")
    args = parser.parse_args()

    seeds = args.urls if args.urls else DEFAULT_SEED_URLS
    crawl(
        seed_urls=seeds,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        same_domain_only=args.same_domain
    )

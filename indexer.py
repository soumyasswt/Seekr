# indexer.py - Seekr Local File Indexer
# Indexes .txt, .md, .html files from docs/ directory

import os
import re
import json
import math
import argparse
import time
from collections import defaultdict

DOCS_PATH  = "docs"
INDEX_FILE = "index.json"
STOPWORDS = {
    "is","a","for","the","and","of","to","in","it","this","that","was","are",
    "be","as","at","by","we","or","an","on","with","you","he","she","they",
    "but","from","has","had","have","not","their","its","if","do","did","so",
    "can","will","all","been","more","also","about","into","than","then",
    "there","when","which","would","could","should","one","two","may","each",
    "other","such","after","before","between","through","during","over"
}
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".rst"}


def tokenize(text: str) -> list:
    text = text.lower()
    # Strip HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    words = re.findall(r"\b[a-z']+\b", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def extract_title(filename: str, content: str) -> str:
    """Try to extract a meaningful title from the content."""
    # Check for markdown heading
    m = re.match(r"#\s+(.+)", content)
    if m:
        return m.group(1).strip()
    # First non-empty line
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return filename


def index_documents(docs_path: str = DOCS_PATH, output_file: str = INDEX_FILE):
    inverted_index = defaultdict(dict)  # {token: {filename: freq}}
    doc_titles     = {}
    doc_snippets   = {}
    total_files    = 0

    print(f"[Seekr Indexer] Scanning {docs_path}/")

    for root, dirs, files in os.walk(docs_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, start=".").replace("\\", "/")

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                print(f"  Skipping {filepath}: {e}")
                continue

            tokens = tokenize(content)
            if not tokens:
                continue

            total_files += 1
            doc_titles[rel_path]   = extract_title(filename, content)
            doc_snippets[rel_path] = content[:300].replace("\n", " ").strip()

            # Term frequency
            for token in tokens:
                inverted_index[token][rel_path] = inverted_index[token].get(rel_path, 0) + 1

            print(f"  Indexed: {rel_path} ({len(tokens)} tokens)")

    # Save index
    data = {
        "index":    dict(inverted_index),
        "titles":   doc_titles,
        "snippets": doc_snippets,
        "meta": {
            "total_docs":  total_files,
            "total_terms": len(inverted_index),
            "indexed_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[Seekr Indexer] Done!")
    print(f"  Files indexed : {total_files}")
    print(f"  Unique terms  : {len(inverted_index)}")
    print(f"  Output        : {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seekr Local File Indexer")
    parser.add_argument("--docs",   default=DOCS_PATH,  help="Directory to index")
    parser.add_argument("--output", default=INDEX_FILE, help="Output index file")
    args = parser.parse_args()

    os.makedirs(args.docs, exist_ok=True)
    index_documents(args.docs, args.output)

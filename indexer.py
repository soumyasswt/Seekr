import os
import re
import json
from collections import defaultdict

DOCS_PATH = "docs"
INDEX_FILE = "index.json"
STOPWORDS = {"is", "a", "for", "the", "and", "of", "to", "in"}

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS]

# New: store {token: {doc: frequency}}
inverted_index = defaultdict(dict)

for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".txt"):
        with open(os.path.join(DOCS_PATH, filename), "r", encoding="utf-8") as f:
            tokens = tokenize(f.read())
            for token in tokens:
                if filename in inverted_index[token]:
                    inverted_index[token][filename] += 1
                else:
                    inverted_index[token][filename] = 1

with open(INDEX_FILE, "w") as f:
    json.dump(inverted_index, f, indent=2)

print("Index built and saved to index.json")
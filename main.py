import os
import re
from collections import defaultdict

DOCS_PATH = "docs"
STOPWORDS = {"is", "a", "for", "the", "and", "of", "to", "in"}

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS]

# -------- INDEXING --------
inverted_index = defaultdict(list)

for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".txt"):
        with open(os.path.join(DOCS_PATH, filename), "r", encoding="utf-8") as f:
            tokens = tokenize(f.read())
            for token in tokens:
                inverted_index[token].append(filename)

# -------- SEARCH --------
query = input("Search: ")
q_tokens = tokenize(query)

scores = {}

for token in q_tokens:
    docs = inverted_index.get(token, [])
    for doc in docs:
        scores[doc] = scores.get(doc, 0) + (1 / len(inverted_index[token]))

final_docs = []
for doc in scores:
    if all(doc in inverted_index.get(token, []) for token in q_tokens):
        final_docs.append((doc, scores[doc]))

final_docs.sort(key=lambda x: x[1], reverse=True)

if final_docs:
    print("Results:", [doc for doc, score in final_docs])
else:
    print("No results found.")
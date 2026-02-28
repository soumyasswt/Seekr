# backend.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
import re

INDEX_FILE = "web_index.json"
STOPWORDS = {"is", "a", "for", "the", "and", "of", "to", "in"}

app = FastAPI(title="Mini Search Engine")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Load index once at startup
with open(INDEX_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
    inverted_index = data["index"]
    page_titles = data["titles"]

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS]

def search_query(query: str):
    q_tokens = set(tokenize(query))
    doc_scores = {}
    doc_matches = {}

    for token in q_tokens:
        docs_freq = inverted_index.get(token, {})
        rarity = 1 / len(docs_freq) if docs_freq else 0
        for doc, freq in docs_freq.items():
            doc_scores[doc] = doc_scores.get(doc, 0) + freq * rarity
            doc_matches.setdefault(doc, set()).add(token)

    # Sort by matched tokens count and score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: (-len(doc_matches[x[0]]), -x[1]))

    results = []
    for doc, score in sorted_docs:
        matched_tokens = doc_matches[doc]
        top_token = max(matched_tokens, key=lambda t: inverted_index[t][doc] * (1 / len(inverted_index[t])))
        results.append({
            "url": doc,
            "title": page_titles.get(doc, doc),
            "matched_tokens": list(matched_tokens),
            "top_token": top_token
        })
    return results

@app.get("/search")
def search(q: str = Query(..., description="Search query")):
    results = search_query(q)
    return {"query": q, "results": results[:20]}  # limit to top 20 results
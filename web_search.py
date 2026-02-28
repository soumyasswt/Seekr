# web_search.py
import json
import re

INDEX_FILE = "web_index.json"
STOPWORDS = {"is", "a", "for", "the", "and", "of", "to", "in"}

def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["index"], data["titles"]

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS]

def search_query(query, inverted_index):
    q_tokens = set(tokenize(query))
    doc_scores = {}
    doc_matches = {}

    for token in q_tokens:
        docs_freq = inverted_index.get(token, {})
        rarity = 1 / len(docs_freq) if docs_freq else 0
        for doc, freq in docs_freq.items():
            doc_scores[doc] = doc_scores.get(doc, 0) + freq * rarity
            doc_matches.setdefault(doc, set()).add(token)

    # Sort by number of matched tokens then score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: (-len(doc_matches[x[0]]), -x[1]))

    results = []
    for doc, score in sorted_docs:
        matched_tokens = doc_matches[doc]
        top_token = max(matched_tokens, key=lambda t: inverted_index[t][doc] * (1 / len(inverted_index[t])))
        results.append((doc, matched_tokens, top_token))
    return results

def main():
    inverted_index, page_titles = load_index()
    print("Enter queries one per line. Type 'EXIT' to quit.")
    while True:
        query = input("Search: ").strip()
        if query.lower() == "exit":
            break
        results = search_query(query, inverted_index)
        if results:
            print("Results:")
            for url, matched_tokens, top_token in results:
                title = page_titles.get(url, url)
                print(f"  {title} ({url}) - matched: {', '.join(matched_tokens)}; top: {top_token})")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
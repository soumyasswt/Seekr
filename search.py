import json
import re
import os

INDEX_FILE = "index.json"
DOCS_PATH = "docs"
STOPWORDS = {"is", "a", "for", "the", "and", "of", "to", "in"}

# ---------- UTILITIES ----------
def load_index():
    with open(INDEX_FILE, "r") as f:
        return json.load(f)

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return [w for w in words if w not in STOPWORDS]

def load_docs_content():
    """Load full text of all documents for phrase search."""
    content = {}
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, filename), "r", encoding="utf-8") as f:
                content[filename] = f.read()
    return content

# ---------- PHRASE SEARCH ----------
def phrase_in_doc(phrase, text):
    return phrase.lower() in text.lower()

# ---------- QUERY PARSING ----------
def parse_query(query):
    """Return tokens: quoted phrases, operators, parentheses, or words."""
    pattern = r'"[^"]+"|\bAND\b|\bOR\b|\bNOT\b|\(|\)|\b[a-zA-Z]+\b'
    return re.findall(pattern, query, flags=re.IGNORECASE)

# ---------- BOOLEAN EVALUATION ----------
def eval_query(tokens, inverted_index, all_docs, docs_content):
    """Evaluate query recursively with proper boolean precedence."""
    if not tokens:
        return set()
    
    def helper(tokens):
        res_stack = []
        op_stack = []

        def apply_op():
            if not op_stack:
                return
            op = op_stack.pop()
            if op == "AND":
                b = res_stack.pop()
                a = res_stack.pop()
                res_stack.append(a & b)
            elif op == "OR":
                b = res_stack.pop()
                a = res_stack.pop()
                res_stack.append(a | b)
            elif op == "NOT":
                a = res_stack.pop()
                res_stack.append(all_docs - a)

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            tok_upper = tok.upper()

            if tok_upper in {"AND", "OR"}:
                while op_stack and op_stack[-1] == "AND" and tok_upper == "OR":
                    apply_op()
                op_stack.append(tok_upper)
                i += 1
            elif tok_upper == "NOT":
                op_stack.append("NOT")
                i += 1
            elif tok == "(":
                depth = 1
                j = i + 1
                while j < len(tokens) and depth > 0:
                    if tokens[j] == "(":
                        depth += 1
                    elif tokens[j] == ")":
                        depth -= 1
                    j += 1
                res_stack.append(helper(tokens[i+1:j-1]))
                i = j
            elif tok.startswith('"') and tok.endswith('"'):
                phrase = tok.strip('"')
                matched_docs = {doc for doc, text in docs_content.items() if phrase_in_doc(phrase, text)}
                res_stack.append(matched_docs)
                i += 1
            else:
                res_stack.append(set(inverted_index.get(tok.lower(), {}).keys()))
                i += 1

        while op_stack:
            apply_op()
        return res_stack[0] if res_stack else set()

    return helper(tokens)

# ---------- SEARCH FUNCTION ----------
def search_query(query, inverted_index, docs_content):
    all_docs = set(docs_content.keys())
    tokens = parse_query(query)
    
    # If no boolean operators, default OR between words
    has_boolean = any(tok.upper() in {"AND", "OR", "NOT"} for tok in tokens)
    if not has_boolean:
        tokens_with_or = []
        for tok in tokens:
            if tok.startswith('"') and tok.endswith('"'):
                tokens_with_or.append(tok)
            else:
                tokens_with_or.append(tok)
                tokens_with_or.append("OR")
        if tokens_with_or:
            tokens_with_or.pop()  # Remove last OR
        tokens = tokens_with_or

    matched_docs = eval_query(tokens, inverted_index, all_docs, docs_content)

    # Score and top token
    doc_scores = {}
    doc_matches = {}
    q_tokens = [t.lower().strip('"') for t in tokens if t.isalpha() or t.startswith('"')]
    for token in q_tokens:
        if token.startswith('"'):
            phrase = token.strip('"')
            for doc in matched_docs:
                if phrase_in_doc(phrase, docs_content[doc]):
                    doc_scores[doc] = doc_scores.get(doc, 0) + len(phrase.split())
                    doc_matches.setdefault(doc, set()).add(phrase)
        else:
            docs_freq = inverted_index.get(token, {})
            rarity = 1 / len(docs_freq) if docs_freq else 1
            for doc in matched_docs:
                if doc in docs_freq:
                    score = docs_freq[doc] * rarity
                    doc_scores[doc] = doc_scores.get(doc, 0) + score
                    doc_matches.setdefault(doc, set()).add(token)

    results = []
    for doc in matched_docs:
        matched_tokens = doc_matches.get(doc, set())
        top_token = max(matched_tokens, key=lambda t: doc_scores.get(doc, 0)) if matched_tokens else ""
        results.append((doc, matched_tokens, top_token))

    # Sort by number of matches then score
    results.sort(key=lambda x: (-len(x[1]), -doc_scores.get(x[0], 0)))
    return results

# ---------- MAIN ----------
def main():
    inverted_index = load_index()
    docs_content = load_docs_content()
    print("Enter queries one per line. Type 'EXIT' to quit.")
    while True:
        query = input("Search: ").strip()
        if query.lower() == "exit":
            break
        results = search_query(query, inverted_index, docs_content)
        if results:
            print("Results:")
            for doc, matched_tokens, top_token in results:
                if matched_tokens:
                    print(f"  {doc} (matched: {', '.join(matched_tokens)}; top: {top_token})")
                else:
                    print(f"  {doc}")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()
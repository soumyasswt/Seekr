# Seekr 🔍 — Mini Google Search Engine

A production-ready, self-hosted search engine with BM25 ranking, web crawling, spell correction, and a beautiful UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| **BM25 Ranking** | Beats TF-IDF with length-normalised, frequency-saturated scoring |
| **PageRank** | Iterative PageRank computed from crawled link graphs |
| **Spell Correction** | Edit-distance based correction for misspelled queries |
| **Snippet Extraction** | Highlights matching terms in document excerpts |
| **Autocomplete** | Real-time suggestions as you type |
| **Boolean Search** | Supports `AND`, `OR`, `NOT` operators |
| **Web Crawler** | Respects `robots.txt`, rate-limits, multi-depth crawl |
| **Local Indexing** | Index `.txt`, `.md`, `.html` files from any directory |
| **Pagination** | Fast paginated results with page navigation |
| **REST API** | Clean FastAPI endpoints — easy to integrate |
| **Docker** | One-command deploy |

---

## 🚀 Quick Start

### Option 1: Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Index your local documents
mkdir -p docs
# Drop .txt / .md / .html files into docs/
python indexer.py

# 3. (Optional) Crawl the web
python crawler_indexer.py https://en.wikipedia.org/wiki/Python_(programming_language) --max-pages 100

# 4. Start the backend
python backend.py

# 5. Open in browser
open index.html
# OR visit http://127.0.0.1:8000
```

### Option 2: Docker

```bash
docker-compose up --build
# Visit http://localhost:8000
```

---

## 📁 Project Structure

```
seekr/
├── backend.py          # FastAPI backend — BM25, spell, snippets, API
├── indexer.py          # Local file indexer (.txt/.md/.html → index.json)
├── crawler_indexer.py  # Web crawler + PageRank → web_index.json
├── search.py           # CLI search tool (uses index.json)
├── index.html          # Frontend — beautiful search UI
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── docs/               # Drop your documents here
├── index.json          # Generated: local document index
└── web_index.json      # Generated: web crawl index
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /search?q=...` | GET | Search — returns ranked results |
| `GET /suggest?q=...` | GET | Autocomplete suggestions |
| `GET /spell?q=...` | GET | Spell-check a query |
| `GET /stats` | GET | Index statistics |
| `POST /reload` | POST | Hot-reload index from disk |

### Search Parameters

| Param | Default | Description |
|---|---|---|
| `q` | required | Query string |
| `source` | `all` | `all`, `web`, or `local` |
| `page` | `1` | Page number |
| `per_page` | `10` | Results per page (max 50) |

### Example Response

```json
{
  "results": [
    {
      "url": "https://example.com/page",
      "title": "Example Page",
      "snippet": "...the <mark>query</mark> term appears here...",
      "matched_tokens": ["query", "term"],
      "score": 4.2183,
      "source": "web"
    }
  ],
  "total": 42,
  "page": 1,
  "corrected": null,
  "elapsed": 0.0031
}
```

---

## 🕷️ Crawling

```bash
# Crawl specific URLs
python crawler_indexer.py https://docs.python.org/ https://wikipedia.org/

# Stay on same domain
python crawler_indexer.py https://docs.python.org/ --same-domain

# Limit pages and depth
python crawler_indexer.py https://example.com --max-pages 500 --max-depth 4
```

---

## 🔎 Boolean Search

Use operators in the search box:

```
python AND tutorial
web OR internet
NOT javascript
```

---

## 🚢 Deploying to Production

### Railway / Render / Fly.io

1. Push this folder to a GitHub repo
2. Connect to Railway/Render and point to `Dockerfile`
3. Set port to `8000`
4. Done!

### Nginx + Let's Encrypt

```nginx
server {
    listen 443 ssl;
    server_name search.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}
```

Then update `API` in `index.html` from `http://127.0.0.1:8000` to your domain.

---

## ⚙️ Ranking Algorithm

Seekr uses **BM25** (Best Match 25) — the same family of algorithms used by Elasticsearch and Solr:

```
score(D, Q) = Σ IDF(qi) × TF(qi, D) × (k1 + 1) / (TF(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

- **k1 = 1.5** — controls term frequency saturation
- **b = 0.75** — controls document length normalisation
- **PageRank boost** — web results boosted by `1 + 0.3 × log(1 + PR)`
- **Coverage boost** — documents matching more query terms get a multiplicative boost

---

## 🧪 Testing

```bash
# Test the API directly
curl "http://localhost:8000/search?q=python+programming"
curl "http://localhost:8000/suggest?q=prog"
curl "http://localhost:8000/spell?q=progrming"
curl "http://localhost:8000/stats"
```

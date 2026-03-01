# backend.py - Seekr Search Engine Backend v3.1
# Fixes: CORS (no credentials), DDG lite endpoint, /wake for Render cold-start, retries

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json, re, os, math, time, hashlib, threading, random, smtplib, secrets
from collections import defaultdict
import urllib.request, urllib.parse, urllib.error
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ─── CONFIG ──────────────────────────────────────────────────────────────────
LOCAL_INDEX_FILE = "index.json"
CACHE_TTL        = 600
BM25_K1, BM25_B  = 1.5, 0.75
BOOT_TIME        = time.time()

STOPWORDS = {
    "is","a","for","the","and","of","to","in","it","this","that","was","are",
    "be","as","at","by","we","or","an","on","with","you","he","she","they",
    "but","from","has","had","have","not","their","its","if","do","did","so",
    "can","will","all","been","more","also","about","into","than","then",
    "there","when","which","would","could","should","one","two","may","each",
    "other","such","after","before","between","through","during","over",
    "used","use","using","uses","our","up","out","time","year","some","new",
    "no","just","like","him","his","her","my","your","who","what","how",
    "very","get","both","own","same","first","last","long","great","little",
    "even","back","still","way","take","come","since","against","much","most"
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

def _headers():
    return {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection":      "keep-alive",
        "DNT":             "1",
    }

# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Seekr", version="3.1")

# ━━ CORS FIX ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The old config had allow_credentials=True + specific origins. That caused
# browsers to reject the preflight unless every fetch included credentials:'include'.
# Fix: allow_origins=["*"] + allow_credentials=False.
# Auth tokens go in JSON bodies (not cookies), so this is completely safe.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── STATE ───────────────────────────────────────────────────────────────────
state = {
    "local_index": {}, "local_titles": {}, "local_snippets": {},
    "doc_lengths": {}, "avg_doc_len": 1.0, "total_docs": 0, "vocab": set(),
}
_cache: dict     = {}
_cache_lock      = threading.Lock()
_otp_store: dict = {}
_otp_lock        = threading.Lock()


# ─── LOCAL INDEX ─────────────────────────────────────────────────────────────
def load_local_index():
    if not os.path.exists(LOCAL_INDEX_FILE):
        print("[Seekr] No local index — run: python indexer.py")
        return
    with open(LOCAL_INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = data.get("index", data) if isinstance(data, dict) else data
    state["local_index"]    = idx
    state["local_titles"]   = data.get("titles",   {}) if isinstance(data, dict) else {}
    state["local_snippets"] = data.get("snippets", {}) if isinstance(data, dict) else {}
    doc_lens = defaultdict(int)
    for term, postings in idx.items():
        for doc, freq in postings.items():
            doc_lens[doc] += freq
    state["doc_lengths"] = dict(doc_lens)
    count = len(doc_lens)
    state["avg_doc_len"] = sum(doc_lens.values()) / count if count else 1.0
    state["total_docs"]  = count
    state["vocab"]       = set(idx.keys())
    print(f"[Seekr] Local index: {count} docs, {len(idx)} terms.")

load_local_index()


# ─── TEXT UTILS ──────────────────────────────────────────────────────────────
def tokenize(text):
    words = re.findall(r"\b[a-z']+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

def highlight(text, tokens, max_len=280):
    if not text: return ""
    lo   = text.lower()
    best = next((lo.find(t) for t in tokens if lo.find(t) >= 0), 0)
    s    = max(0, best - 60)
    e    = min(len(text), s + max_len)
    snip = ("…" if s else "") + text[s:e] + ("…" if e < len(text) else "")
    for t in tokens:
        snip = re.sub(re.escape(t), lambda m: f"<mark>{m.group()}</mark>", snip, flags=re.IGNORECASE)
    return snip


# ─── SPELL CORRECTION ────────────────────────────────────────────────────────
def _edist(a, b):
    if abs(len(a)-len(b)) > 3: return 99
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            temp = dp[j]
            dp[j] = prev if a[i-1]==b[j-1] else 1+min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def spell_correct_query(query):
    vocab  = state["vocab"]
    tokens = re.findall(r"\b[a-z]+\b", query.lower())
    out, changed = [], False
    for tok in tokens:
        if tok in STOPWORDS or tok in vocab:
            out.append(tok); continue
        cands = [(v, _edist(tok, v)) for v in vocab if abs(len(tok)-len(v)) <= 2 and _edist(tok, v) <= 2]
        cands.sort(key=lambda x: x[1])
        if cands and cands[0][1] > 0:
            out.append(cands[0][0]); changed = True
        else:
            out.append(tok)
    return " ".join(out), changed


# ─── FETCH WITH RETRY ────────────────────────────────────────────────────────
def _fetch(url, timeout=18):
    import gzip
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if "gzip" in resp.info().get("Content-Encoding", ""):
                    raw = gzip.decompress(raw)
                return raw.decode("utf-8", errors="replace")
        except Exception as e:
            if attempt < 2: time.sleep(1.0 * (attempt + 1))
            else: print(f"[Seekr] fetch failed {url}: {e}")
    return ""


# ─── DUCKDUCKGO LITE (cloud-IP friendly) ─────────────────────────────────────
def _extract_ddg_url(href):
    if not href: return ""
    for prefix in ["//duckduckgo.com/l/", "/l/?"]:
        if href.startswith(prefix):
            full = ("https:" + href) if href.startswith("//") else ("https://duckduckgo.com" + href)
            try:
                params = urllib.parse.parse_qs(urllib.parse.urlparse(full).query)
                return urllib.parse.unquote(params.get("uddg", [href])[0])
            except: pass
    return href if href.startswith("http") else ""


def _parse_lite(html):
    results, seen = [], set()
    title_blocks = re.findall(
        r'<a[^>]+class="[^"]*result-link[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html, re.DOTALL | re.IGNORECASE)
    snippets_raw = re.findall(
        r'class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</(?:td|div|span)',
        html, re.DOTALL | re.IGNORECASE)
    display_urls = re.findall(
        r'class="[^"]*result-url[^"]*"[^>]*>(.*?)</(?:span|td)',
        html, re.DOTALL | re.IGNORECASE)
    for i, (href, title_html) in enumerate(title_blocks):
        url   = _extract_ddg_url(href)
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if not url or not title or url in seen or "duckduckgo.com" in url: continue
        seen.add(url)
        snippet = re.sub(r"<[^>]+>", "", snippets_raw[i]).strip() if i < len(snippets_raw) else ""
        display = re.sub(r"<[^>]+>", "", display_urls[i]).strip() if i < len(display_urls) else url
        results.append({"url": url, "title": title, "display_url": display, "snippet": snippet, "source": "web"})
        if len(results) >= 12: break
    return results


def _parse_html_ddg(html):
    results, seen = [], set()
    for block in re.split(r'<div[^>]+class="[^"]*result[^"]*"[^>]*>', html)[1:]:
        m = re.search(r'class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
        if not m: continue
        url   = _extract_ddg_url(m.group(1))
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if not url or not title or url in seen or "duckduckgo.com/y.js" in url: continue
        seen.add(url)
        sm = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:div|span|a)', block, re.DOTALL)
        dm = re.search(r'class="[^"]*result__url[^"]*"[^>]*>(.*?)</(?:a|span)', block, re.DOTALL)
        snippet = re.sub(r"<[^>]+>", "", sm.group(1)).strip() if sm else ""
        display = re.sub(r"<[^>]+>", "", dm.group(1)).strip() if dm else url
        results.append({"url": url, "title": title, "display_url": display, "snippet": snippet, "source": "web"})
        if len(results) >= 12: break
    return results


def fetch_live_results(query, page=1):
    key = hashlib.md5(f"{query.lower()}|{page}".encode()).hexdigest()
    with _cache_lock:
        if key in _cache:
            ts, data = _cache[key]
            if time.time() - ts < CACHE_TTL: return data

    offset  = (page - 1) * 10
    results = []

    # PRIMARY: lite.duckduckgo.com — works on cloud/datacenter IPs
    html = _fetch("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode(
        {"q": query, "kl": "us-en", "s": str(offset) if offset else "0", "dc": str(offset+1)}
    ))
    if html:
        results = _parse_lite(html)
        print(f"[Seekr] lite DDG '{query}' p{page} → {len(results)} results")

    # FALLBACK: html.duckduckgo.com
    if not results:
        extra = {"b": str(offset)} if offset else {}
        html = _fetch("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode(
            {"q": query, "kl": "us-en", "k1": "-1", **extra}
        ))
        if html:
            results = _parse_html_ddg(html)
            print(f"[Seekr] html DDG '{query}' p{page} → {len(results)} results (fallback)")

    if not results:
        print(f"[Seekr] ⚠ No DDG results for '{query}' — both endpoints failed")

    with _cache_lock:
        _cache[key] = (time.time(), results)
    return results


# ─── AUTOCOMPLETE ─────────────────────────────────────────────────────────────
def fetch_suggestions(prefix):
    key = "ac:" + prefix.lower()
    with _cache_lock:
        if key in _cache:
            ts, data = _cache[key]
            if time.time() - ts < 60: return data
    try:
        raw  = _fetch("https://duckduckgo.com/ac/?" + urllib.parse.urlencode({"q": prefix, "type": "list"}), timeout=5)
        data = json.loads(raw) if raw else []
        sug  = data[1] if len(data) > 1 else []
    except:
        sug = [t for t in sorted(state["vocab"]) if t.startswith(prefix.lower())][:8]
    with _cache_lock:
        _cache[key] = (time.time(), sug)
    return sug[:8]


# ─── BM25 ─────────────────────────────────────────────────────────────────────
def bm25_score(term, doc, postings):
    N   = max(state["total_docs"], 1)
    df  = len(postings)
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    tf  = postings.get(doc, 0)
    dl  = state["doc_lengths"].get(doc, state["avg_doc_len"])
    tfn = (tf*(BM25_K1+1)) / (tf + BM25_K1*(1-BM25_B + BM25_B*dl/state["avg_doc_len"]))
    return idf * tfn

def search_local(q_tokens):
    idx = state["local_index"]
    scores, matched = defaultdict(float), defaultdict(set)
    for t in q_tokens:
        for doc in idx.get(t, {}):
            scores[doc]  += bm25_score(t, doc, idx[t])
            matched[doc].add(t)
    results = []
    for doc, score in sorted(scores.items(), key=lambda x: -x[1]):
        cov   = len(matched[doc]) / max(len(q_tokens), 1)
        score *= (0.5 + 0.5 * cov)
        title   = state["local_titles"].get(doc, os.path.basename(doc))
        snippet = state["local_snippets"].get(doc, "")
        if not snippet:
            try:
                p = doc if os.path.isabs(doc) else os.path.join(".", doc)
                if os.path.exists(p):
                    snippet = open(p, encoding="utf-8", errors="ignore").read()
            except: pass
        results.append({
            "url": f"/doc/{doc}", "display_url": doc, "title": title,
            "snippet": highlight(snippet, q_tokens) if snippet else f"Local: {os.path.basename(doc)}",
            "source": "local", "score": round(score, 4), "matched_tokens": list(matched[doc]),
        })
    return results


# ─── MAIN SEARCH ─────────────────────────────────────────────────────────────
def do_search(query, source="all", page=1, per_page=10):
    t0 = time.time()
    corrected, was_corrected = spell_correct_query(query)
    effective = corrected if was_corrected else query
    q_tokens  = tokenize(effective)
    results   = []

    if source in ("all", "web"):
        live = fetch_live_results(effective, page=page)
        for r in live:
            toks = tokenize((r.get("title") or "") + " " + (r.get("snippet") or ""))
            r["matched_tokens"] = list(set(q_tokens) & set(toks)) or q_tokens[:2]
            r["score"] = 1.0
            if r.get("snippet") and q_tokens:
                r["snippet"] = highlight(r["snippet"], q_tokens)
        results.extend(live)

    if source in ("all", "local") and state["local_index"] and q_tokens:
        seen = {r.get("title", "").lower() for r in results}
        for r in search_local(q_tokens):
            if r.get("title", "").lower() not in seen:
                results.append(r)

    if source == "local":
        total   = len(results)
        start   = (page-1)*per_page
        results = results[start:start+per_page]
    else:
        total = max(len(results), per_page * 5)

    return {
        "results": results, "total": total, "page": page, "per_page": per_page,
        "corrected": corrected if was_corrected else None,
        "original_query": query, "elapsed": round(time.time()-t0, 4),
    }


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
@app.get("/wake")
@app.head("/wake")
async def wake():
    """Lightweight ping so the frontend can warm up Render's cold-start."""
    return {"status": "alive", "uptime_s": round(time.time() - BOOT_TIME)}

@app.get("/")
async def root():
    for f in ["index.html", "static/index.html"]:
        if os.path.exists(f): return FileResponse(f)
    return {"message": "Seekr API v3.1"}

@app.get("/search")
async def search_ep(q: str=Query(...), source: str=Query("all"), page: int=Query(1,ge=1), per_page: int=Query(10,ge=1,le=50)):
    if not q.strip(): raise HTTPException(400, "Empty query")
    return do_search(q.strip(), source, page, per_page)

@app.get("/suggest")
async def suggest_ep(q: str=Query(...)):
    return {"suggestions": fetch_suggestions(q.strip())}

@app.get("/spell")
async def spell_ep(q: str=Query(...)):
    c, ch = spell_correct_query(q)
    return {"original": q, "corrected": c, "changed": ch}

@app.get("/stats")
async def stats_ep():
    return {"status": "ok", "local_docs": state["total_docs"],
            "cache_size": len(_cache), "uptime_s": round(time.time()-BOOT_TIME)}

@app.post("/reload")
async def reload_ep():
    load_local_index()
    return {"ok": True, "local_docs": state["total_docs"]}


# ─── AUTH / OTP ───────────────────────────────────────────────────────────────
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)

def _send_otp_email(to_email, otp):
    if not SMTP_USER or not SMTP_PASS:
        print(f"[Seekr Auth] ⚠ No SMTP. OTP for {to_email}: {otp}")
        return True
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{otp} is your Seekr verification code"
        msg["From"]    = f"Seekr <{SMTP_FROM}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(f"""<div style="font-family:'Segoe UI',sans-serif;max-width:480px;margin:0 auto;background:#0e0e10;color:#f0f0f2;border-radius:16px;padding:32px;border:1px solid rgba(255,255,255,0.1)">
          <div style="font-size:28px;font-weight:700;background:linear-gradient(135deg,#7b6ef6,#5b8df6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px">Seekr</div>
          <p style="color:#a8a8b4;font-size:14px;margin-bottom:28px">Your verification code</p>
          <div style="background:#1e1e22;border:1px solid rgba(123,110,246,0.3);border-radius:12px;padding:24px;text-align:center;margin-bottom:24px">
            <span style="font-size:40px;font-weight:700;letter-spacing:0.15em;color:#f0f0f2">{otp}</span>
          </div>
          <p style="color:#6a6a7a;font-size:13px">Expires in <strong style="color:#a8a8b4">10 minutes</strong>. Don't share it.</p>
        </div>""", "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls(); s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_FROM, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[Seekr Auth] Email error: {e}"); return False

@app.post("/auth/send-otp")
async def send_otp(body: dict):
    email = (body.get("email") or "").strip().lower()
    if not email or "@" not in email: raise HTTPException(400, "Invalid email address")
    otp = str(random.randint(100000, 999999))
    with _otp_lock:
        _otp_store[email] = {"otp": otp, "expires": time.time()+600, "attempts": 0}
    if not _send_otp_email(email, otp): raise HTTPException(500, "Failed to send email")
    return {"message": f"OTP sent to {email}"}

@app.post("/auth/verify-otp")
async def verify_otp(body: dict):
    email   = (body.get("email") or "").strip().lower()
    entered = (body.get("otp")   or "").strip()
    with _otp_lock:
        rec = _otp_store.get(email)
        if not rec:                     raise HTTPException(400, "No OTP requested for this email")
        if rec["attempts"] >= 5:
            del _otp_store[email];      raise HTTPException(429, "Too many attempts")
        if time.time() > rec["expires"]:
            del _otp_store[email];      raise HTTPException(400, "OTP expired")
        rec["attempts"] += 1
        if entered != rec["otp"]:       raise HTTPException(400, f"Incorrect OTP. {5-rec['attempts']} attempts left.")
        del _otp_store[email]
    return {"token": secrets.token_hex(32), "email": email, "name": email.split("@")[0].capitalize()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)

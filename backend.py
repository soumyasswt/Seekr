
# backend.py - Seekr Search Engine Backend v3.3 (Web-Only)
# Simplified backend to focus on reliable web search.

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json, re, os, time, hashlib, threading, random, smtplib, secrets
from duckduckgo_search import DDGS
from urllib.parse import urlparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CACHE_TTL = 600
BOOT_TIME = time.time()

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

# ─── APP ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Seekr", version="3.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── STATE ───────────────────────────────────────────────────────────────────
_cache: dict     = {}
_cache_lock      = threading.Lock()
_otp_store: dict = {}
_otp_lock        = threading.Lock()

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

# ─── LIVE WEB SEARCH (duckduckgo-search) ───────────────────────────────────
def fetch_live_results(query, page=1):
    key = hashlib.md5(f"{query.lower()}".encode()).hexdigest()
    with _cache_lock:
        if key in _cache:
            ts, data = _cache[key]
            if time.time() - ts < CACHE_TTL:
                start = (page - 1) * 10
                return data[start:start+10]

    results = []
    try:
        with DDGS(timeout=20) as ddgs:
            raw_results = ddgs.text(query, safesearch='moderate', max_results=25)
            for r in raw_results:
                hostname = urlparse(r.get('href')).hostname if r.get('href') else ''
                results.append({
                    "url": r.get('href'),
                    "title": r.get('title'),
                    "display_url": hostname,
                    "snippet": r.get('body'),
                    "source": "web"
                })
        print(f"[Seekr] DDGS search for '{query}' → {len(results)} results")
    except Exception as e:
        print(f"[Seekr] ⚠ DDGS search failed for '{query}': {e}")
        return []

    with _cache_lock:
        _cache[key] = (time.time(), results)

    return results[:10]

# ─── AUTOCOMPLETE ─────────────────────────────────────────────────────────────
def fetch_suggestions(prefix):
    key = "ac:" + prefix.lower()
    with _cache_lock:
        if key in _cache:
            ts, data = _cache[key]
            if time.time() - ts < 60: return data
    try:
        with DDGS(timeout=5) as ddgs:
            sug = [r['phrase'] for r in ddgs.suggestions(prefix, max_results=8)]
    except Exception as e:
        print(f"[Seekr] AC failed: {e}")
        sug = [] # No local vocab to fall back on

    with _cache_lock:
        _cache[key] = (time.time(), sug)
    return sug[:8]

# ─── MAIN SEARCH ─────────────────────────────────────────────────────────────
def do_search(query, page=1, per_page=10):
    t0 = time.time()
    q_tokens  = tokenize(query)
    results   = []

    live = fetch_live_results(query, page=page)
    for r in live:
        toks = tokenize((r.get("title") or "") + " " + (r.get("snippet") or ""))
        r["matched_tokens"] = list(set(q_tokens) & set(toks)) or q_tokens[:2]
        r["score"] = 1.0
        if r.get("snippet") and q_tokens:
            r["snippet"] = highlight(r["snippet"], q_tokens)
    results.extend(live)

    total = len(results) + (10 if len(results) == 10 else 0)

    return {
        "results": results, "total": total, "page": page, "per_page": per_page,
        "corrected": None, "original_query": query, "elapsed": round(time.time()-t0, 4),
    }

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
@app.get("/wake")
@app.head("/wake")
async def wake():
    return {"status": "alive", "uptime_s": round(time.time() - BOOT_TIME)}

@app.get("/")
async def root():
    for f in ["index.html", "static/index.html"]:
        if os.path.exists(f): return FileResponse(f)
    return {"message": "Seekr API v3.3"}

@app.get("/search")
async def search_ep(q: str=Query(...), page: int=Query(1,ge=1), per_page: int=Query(10,ge=1,le=50)):
    if not q.strip(): raise HTTPException(400, "Empty query")
    return do_search(q.strip(), page, per_page)

@app.get("/suggest")
async def suggest_ep(q: str=Query(...)):
    return {"suggestions": fetch_suggestions(q.strip())}

@app.get("/stats")
async def stats_ep():
    return {"status": "ok", "local_docs": 0,
            "cache_size": len(_cache), "uptime_s": round(time.time()-BOOT_TIME)}

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
        msg.attach(MIMEText(f'''<div style="font-family:'Segoe UI',sans-serif;max-width:480px;margin:0 auto;background:#0e0e10;color:#f0f0f2;border-radius:16px;padding:32px;border:1px solid rgba(255,255,255,0.1)">
          <div style="font-size:28px;font-weight:700;background:linear-gradient(135deg,#7b6ef6,#5b8df6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px">Seekr</div>
          <p style="color:#a8a8b4;font-size:14px;margin-bottom:28px">Your verification code</p>
          <div style="background:#1e1e22;border:1px solid rgba(123,110,246,0.3);border-radius:12px;padding:24px;text-align:center;margin-bottom:24px">
            <span style="font-size:40px;font-weight:700;letter-spacing:0.15em;color:#f0f0f2">{otp}</span>
          </div>
          <p style="color:#6a6a7a;font-size:13px">Expires in <strong style="color:#a8a8b4">10 minutes</strong>. Don't share it.</p>
        </div>''', "html"))
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

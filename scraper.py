"""
scraper.py
Two-layer scraping strategy:
  Layer 1: ScraperAPI (parallel, handles most sites)
  Layer 2: Playwright fallback via subprocess (Windows/Streamlit safe)

The Playwright layer calls playwright_worker.py as a child process so
it gets its own event loop — no asyncio conflicts with Streamlit/Windows.

Per-domain enhancements added for:
  - askmap.net          (static, custom CSS selectors)
  - brownbook.net       (JS-render + geo/cookie headers)
  - freelistingusa.com  (static, thin-page threshold override)
  - hotfrog.com         (heavy SPA — render=true, premium, long wait)
  - nearfinderus.com    (JS-render, already handled — tuned wait)
  - smallbusinessusa.com (moderate JS — render=true, wait for .business-info)
  - us.enrollbusiness.com (JS-render, already handled — tuned wait)
"""

import re
import time
import concurrent.futures
import requests
from bs4 import BeautifulSoup


SCRAPERAPI_ENDPOINT = "http://api.scraperapi.com"

# ── Per-domain render strategy ────────────────────────────────────────────────

# Domains that must use render=true on the FIRST attempt (JS-heavy SPAs)
JS_RENDER_FIRST_DOMAINS = [
    "nearfinderus.com",
    "us.enrollbusiness.com",
    "hotfrog.com",
    "smallbusinessusa.com",
    "brownbook.net",
]

# Domains where render=false is sufficient (static HTML)
STATIC_DOMAINS = [
    "askmap.net",
    "freelistingusa.com",
]

# Per-domain render settings: wait (ms) and CSS selector to wait for
JS_RENDER_CONFIG = {
    "nearfinderus.com": {
        "wait": "20000",
        "wait_for_selector": "h1",
    },
    "us.enrollbusiness.com": {
        "wait": "18000",
        "wait_for_selector": "h1",
    },
    "hotfrog.com": {
        "wait": "20000",
        "wait_for_selector": "h1",
        "premium_first": True,   # force premium on first render attempt
    },
    "smallbusinessusa.com": {
        "wait": "15000",
        "wait_for_selector": "body",
    },
    "brownbook.net": {
        "wait": "15000",
        "wait_for_selector": "h1",
    },
}

# Per-domain minimum content thresholds (chars).
# Some directory pages are intentionally sparse but still valid.
DOMAIN_MIN_CHARS = {
    "askmap.net":        150,
    "freelistingusa.com": 100,
    "smallbusinessusa.com": 150,
}

# Per-domain custom HTTP headers forwarded via ScraperAPI keep_headers
DOMAIN_HEADERS = {
    "brownbook.net": {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
}


# ── Attempt order ─────────────────────────────────────────────────────────────

def _get_attempt_order(url: str) -> list:
    """Return scrape attempt list for a URL.

    Rules (checked in order):
      1. Static-only domains  → render=false only
      2. JS-render-first      → render=true (possibly premium first), then fallbacks
      3. Default              → render=false first, then render=true, then premium
    """
    url_lower = url.lower()

    if any(d in url_lower for d in STATIC_DOMAINS):
        return [
            dict(render=False, premium=False, label="render=false"),
        ]

    if any(d in url_lower for d in JS_RENDER_FIRST_DOMAINS):
        domain_cfg = next(
            (JS_RENDER_CONFIG[d] for d in JS_RENDER_CONFIG if d in url_lower), {}
        )
        premium_first = domain_cfg.get("premium_first", False)
        if premium_first:
            return [
                dict(render=True, premium=True,  label="render=true+premium"),
                dict(render=True, premium=False, label="render=true"),
                dict(render=False, premium=False, label="render=false"),
            ]
        return [
            dict(render=True,  premium=False, label="render=true"),
            dict(render=True,  premium=True,  label="render=true+premium"),
            dict(render=False, premium=False, label="render=false"),
        ]

    # Default fallback order
    return [
        dict(render=False, premium=False, label="render=false"),
        dict(render=True,  premium=False, label="render=true"),
        dict(render=True,  premium=True,  label="render=true+premium"),
    ]


# ── Block-signal detection ────────────────────────────────────────────────────

BLOCK_SIGNALS = [
    "captcha",
    "are you human",
    "cf-browser-verification",
    "ddos-guard",
    "checking your browser",
    "verify you are human",
    "enable cookies to continue",
    "please enable cookies",
    "security check",
    "access to this page has been denied",
    "403 forbidden",
    "robot or human",
    "i am not a robot",
    "please verify",
]

# Signals that indicate a page returned the site's home/listing page rather
# than the actual business profile (redirect to root on unknown slug).
REDIRECT_SIGNALS = {
    "freelistingusa.com":  ["search results", "find local business", "add your business free"],
    "smallbusinessusa.com": ["small business directory", "find a business", "browse categories"],
    "askmap.net":          ["search for places", "add your business"],
    "hotfrog.com":         ["find a", "search results", "business directory"],
    "brownbook.net":       ["business directory", "find businesses"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_blocked(html: str, text: str) -> bool:
    combined = (html[:3000] + text[:1000]).lower()
    return any(signal in combined for signal in BLOCK_SIGNALS)


def _is_redirected(url: str, text: str) -> bool:
    """Detect when the site silently redirected to its home/search page."""
    url_lower = url.lower()
    text_lower = text[:2000].lower()
    for domain, signals in REDIRECT_SIGNALS.items():
        if domain in url_lower:
            hits = sum(1 for s in signals if s in text_lower)
            if hits >= 2:
                return True
    return False


def _min_chars_for(url: str, default: int = 200) -> int:
    url_lower = url.lower()
    for domain, threshold in DOMAIN_MIN_CHARS.items():
        if domain in url_lower:
            return threshold
    return default


def _is_thin(text: str, url: str = "") -> bool:
    return len(text.strip()) < _min_chars_for(url)


def _parse(html: str) -> tuple:
    """Extract clean text and title from raw HTML.

    Keeps header/nav elements because some directories (e.g. nearfinderus,
    askmap) embed the business name / key info inside those elements.
    Strips script, style, noscript, iframe, svg as before.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text  = soup.get_text(separator="\n", strip=True)
    text  = re.sub(r"\n{3,}", "\n\n", text)
    return text, title


def _domain_extra_headers(url: str) -> dict:
    url_lower = url.lower()
    for domain, headers in DOMAIN_HEADERS.items():
        if domain in url_lower:
            return headers
    return {}


# ── Layer 1: ScraperAPI ───────────────────────────────────────────────────────

def _scraperapi_fetch(url: str, api_key: str, render: bool,
                      premium: bool = False, timeout: int = 90):
    from urllib.parse import quote, urlencode

    # Manually encode the target URL to preserve special chars like +
    encoded_url = quote(url, safe="")
    qs = {
        "api_key":      api_key,
        "render":       "true" if render else "false",
        "country_code": "us",
        "keep_headers": "true",
    }

    if render:
        url_lower = url.lower()
        domain_cfg = next(
            (cfg for domain, cfg in JS_RENDER_CONFIG.items() if domain in url_lower),
            {"wait": "12000", "wait_for_selector": "body"},
        )
        qs["wait_for_selector"] = domain_cfg["wait_for_selector"]
        qs["wait"]              = domain_cfg["wait"]
        timeout = max(timeout, int(domain_cfg["wait"]) // 1000 + 30)

    if premium:
        qs["premium"] = "true"

    # Inject extra domain-specific headers when ScraperAPI keep_headers is on
    extra_headers = _domain_extra_headers(url)
    full_url = f"{SCRAPERAPI_ENDPOINT}?{urlencode(qs)}&url={encoded_url}"

    try:
        return requests.get(full_url, headers=extra_headers, timeout=timeout)
    except Exception:
        return None


def _scrape_via_scraperapi(url: str, api_key: str) -> dict:
    debug = ""
    attempts = _get_attempt_order(url)

    for attempt in attempts:
        resp = _scraperapi_fetch(url, api_key,
                                 render=attempt["render"],
                                 premium=attempt["premium"])
        if resp is None:
            debug += f"[{attempt['label']}] timeout\n"
            time.sleep(2)
            continue

        if resp.status_code == 401:
            debug += "HTTP 401 — invalid ScraperAPI key\n"
            break

        if resp.status_code != 200:
            debug += f"[{attempt['label']}] HTTP {resp.status_code}\n"
            if resp.status_code == 429:
                time.sleep(5)
            continue

        html = resp.text
        text, title = _parse(html)
        debug += f"[{attempt['label']}] HTTP 200 | text={len(text):,} chars\n"

        if _is_blocked(html, text):
            debug += "  -> blocked/CAPTCHA\n"
            time.sleep(1)
            continue

        if _is_thin(text, url):
            debug += f"  -> too thin ({len(text.strip())} chars)\n"
            time.sleep(1)
            continue

        if _is_redirected(url, text):
            debug += "  -> silently redirected to home/search page\n"
            time.sleep(1)
            continue

        return {"success": True, "html": html, "text": text,
                "title": title, "debug": debug + "  -> ScraperAPI OK"}

    return {"success": False, "debug": debug}


# ── Layer 2: Playwright (Windows-safe via subprocess) ────────────────────────

def _scrape_via_playwright(url: str, timeout_ms: int = 45000) -> dict:
    """
    Calls playwright_worker.py as a subprocess to completely avoid the
    Windows asyncio / Streamlit ProactorEventLoop conflict.
    The worker runs its own event loop in its own process — no shared state.
    """
    import subprocess
    import json as _json
    import sys
    import os

    # Playwright gets more time on JS-heavy domains
    url_lower = url.lower()
    for domain, cfg in JS_RENDER_CONFIG.items():
        if domain in url_lower:
            timeout_ms = max(timeout_ms, int(cfg["wait"]) + 10000)
            break

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "playwright_worker.py")
    if not os.path.exists(worker):
        return {
            "success": False,
            "debug": f"playwright_worker.py not found at {worker}",
        }

    try:
        proc = subprocess.run(
            [sys.executable, worker, url, str(timeout_ms)],
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout.strip()
        if not stdout:
            stderr_snippet = proc.stderr.strip()[:300] if proc.stderr else "no output"
            return {
                "success": False,
                "debug": f"Playwright worker produced no output. stderr: {stderr_snippet}",
            }
        result = _json.loads(stdout)
        # Apply the same redirect guard to Playwright results too
        if result.get("success") and _is_redirected(url, result.get("text", "")):
            result["success"] = False
            result["debug"]   = (result.get("debug", "") +
                                  "\n  -> Playwright: silently redirected to home/search page")
        return result
    except subprocess.TimeoutExpired:
        return {"success": False, "debug": "Playwright worker timed out (120s)"}
    except _json.JSONDecodeError as e:
        return {"success": False, "debug": f"Playwright worker bad JSON: {e}"}
    except Exception as e:
        return {"success": False, "debug": f"Playwright subprocess error: {e}"}


# ── Combined scraper ──────────────────────────────────────────────────────────

def scrape_page(url: str, api_key: str) -> dict:
    """
    Scrape one URL: ScraperAPI first, Playwright fallback if needed.
    Returns: {url, html, text, title, error, _debug}
    """
    result = {"url": url, "html": "", "text": "", "title": "",
              "error": None, "_debug": ""}

    # Layer 1: ScraperAPI
    sa = _scrape_via_scraperapi(url, api_key)
    result["_debug"] += "[ScraperAPI]\n" + sa.get("debug", "") + "\n"

    if sa["success"]:
        result.update({"html": sa["html"], "text": sa["text"],
                        "title": sa["title"]})
        return result

    # Layer 2: Playwright
    result["_debug"] += "\n[Playwright fallback]\n"
    pw = _scrape_via_playwright(url)
    result["_debug"] += pw.get("debug", "") + "\n"

    if pw["success"]:
        result.update({"html": pw["html"], "text": pw["text"],
                        "title": pw["title"]})
        return result

    result["error"] = (
        "Both ScraperAPI and Playwright failed to retrieve usable content. "
        "See debug info in the app for details."
    )
    return result


def scrape_batch(urls: list, api_key: str, batch_size: int = 5) -> list:
    """
    Phase 1 — all URLs via ScraperAPI in parallel.
    Phase 2 — failed URLs via Playwright sequentially (one browser at a time).
    """
    sa_results = [None] * len(urls)

    # Phase 1: ScraperAPI (parallel)
    def fetch_sa(index_url):
        idx, url = index_url
        return idx, _scrape_via_scraperapi(url, api_key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(fetch_sa, (i, url)): i
                   for i, url in enumerate(urls)}
        for future in concurrent.futures.as_completed(futures):
            idx, sa = future.result()
            sa_results[idx] = sa

    # Phase 2: Playwright for ScraperAPI failures (sequential)
    results = []
    for idx, (url, sa) in enumerate(zip(urls, sa_results)):
        base_debug = "[ScraperAPI]\n" + sa.get("debug", "") + "\n"

        if sa["success"]:
            results.append({
                "url": url, "html": sa["html"], "text": sa["text"],
                "title": sa["title"], "error": None,
                "_debug": base_debug + "-> ScraperAPI succeeded",
            })
        else:
            base_debug += "\n[Playwright fallback]\n"
            pw = _scrape_via_playwright(url)
            base_debug += pw.get("debug", "") + "\n"

            if pw["success"]:
                results.append({
                    "url": url, "html": pw["html"], "text": pw["text"],
                    "title": pw["title"], "error": None,
                    "_debug": base_debug + "-> Playwright succeeded",
                })
            else:
                results.append({
                    "url": url, "html": "", "text": "", "title": "",
                    "error": "Both ScraperAPI and Playwright failed. Check debug info.",
                    "_debug": base_debug,
                })

    return results


# ── Text cleaners ─────────────────────────────────────────────────────────────

def clean_html(html: str, max_chars: int = 60000) -> str:
    # Remove non-content tags — but NOT svg inside <img> or picture tags
    for tag in ["script", "style", "head", "noscript", "iframe"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                      flags=re.DOTALL | re.IGNORECASE)
    # Remove standalone <svg> blocks (icons) but preserve <img> tags entirely
    html = re.sub(r"<svg(?:\s[^>]*)?>.*?</svg>", "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    html = re.sub(r"\s+", " ", html)
    return html[:max_chars]


def clean_text(text: str, max_chars: int = 30000) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()[:max_chars]

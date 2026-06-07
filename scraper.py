"""
scraper.py
Two-layer scraping strategy:
  Layer 1: ScraperAPI (parallel, handles most sites)
  Layer 2: Playwright fallback via subprocess (Windows/Streamlit safe)

nearfinderus fix: description, hours, website, social all live in JS-rendered
sections that need extra wait time and scroll. We now wait for the
"More about" section and the hours table to appear before capturing HTML.
"""

import re
import time
import concurrent.futures
import requests
from bs4 import BeautifulSoup


SCRAPERAPI_ENDPOINT = "http://api.scraperapi.com"

# ── Per-domain render strategy ────────────────────────────────────────────────

JS_RENDER_FIRST_DOMAINS = [
    "nearfinderus.com",
    "us.enrollbusiness.com",
    "hotfrog.com",
    "smallbusinessusa.com",
    "brownbook.net",
]

STATIC_DOMAINS = [
    "askmap.net",
    "freelistingusa.com",
]

JS_RENDER_CONFIG = {
    "nearfinderus.com": {
        # Longer wait — description & hours render late via JS
        "wait": "25000",
        "wait_for_selector": "h1",
        # Extra Playwright timeout for this domain (ms)
        "playwright_timeout": 55000,
    },
    "us.enrollbusiness.com": {
        "wait": "20000",
        "wait_for_selector": "h1",
        "playwright_timeout": 50000,
    },
    "hotfrog.com": {
        "wait": "22000",
        "wait_for_selector": "h1",
        "premium_first": True,
        "playwright_timeout": 55000,
    },
    "smallbusinessusa.com": {
        "wait": "15000",
        "wait_for_selector": "body",
        "playwright_timeout": 45000,
    },
    "brownbook.net": {
        "wait": "15000",
        "wait_for_selector": "h1",
        "playwright_timeout": 45000,
    },
}

DOMAIN_MIN_CHARS = {
    "askmap.net":          150,
    "freelistingusa.com":  100,
    "smallbusinessusa.com": 150,
}

DOMAIN_HEADERS = {
    "brownbook.net": {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
}


# ── Attempt order ─────────────────────────────────────────────────────────────

def _get_attempt_order(url: str) -> list:
    url_lower = url.lower()

    if any(d in url_lower for d in STATIC_DOMAINS):
        return [dict(render=False, premium=False, label="render=false")]

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

    return [
        dict(render=False, premium=False, label="render=false"),
        dict(render=True,  premium=False, label="render=true"),
        dict(render=True,  premium=True,  label="render=true+premium"),
    ]


# ── Block-signal detection ────────────────────────────────────────────────────

BLOCK_SIGNALS = [
    "captcha", "are you human", "cf-browser-verification",
    "ddos-guard", "checking your browser", "verify you are human",
    "enable cookies to continue", "please enable cookies",
    "security check", "access to this page has been denied",
    "403 forbidden", "robot or human", "i am not a robot", "please verify",
]

REDIRECT_SIGNALS = {
    "freelistingusa.com":   ["search results", "find local business", "add your business free"],
    "smallbusinessusa.com": ["small business directory", "find a business", "browse categories"],
    "askmap.net":           ["search for places", "add your business"],
    "hotfrog.com":          ["find a", "search results", "business directory"],
    "brownbook.net":        ["business directory", "find businesses"],
}


def _is_blocked(html: str, text: str) -> bool:
    combined = (html[:3000] + text[:1000]).lower()
    return any(signal in combined for signal in BLOCK_SIGNALS)


def _is_redirected(url: str, text: str) -> bool:
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


# ── nearfinderus content quality check ───────────────────────────────────────

def _nearfinderus_has_description(text: str) -> bool:
    """
    Returns True when the scraped text contains the description section.
    nearfinderus renders 'More about X in City' + description text via JS.
    If this section is absent the scrape is incomplete.
    """
    tl = text.lower()
    return (
        "more about" in tl
        or "opening hours" in tl
        or (len(text) > 3000 and "water damage" in tl)  # generic content check
    )


# ── Layer 1: ScraperAPI ───────────────────────────────────────────────────────

def _scraperapi_fetch(url: str, api_key: str, render: bool,
                      premium: bool = False, timeout: int = 90):
    from urllib.parse import quote, urlencode

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
        timeout = max(timeout, int(domain_cfg["wait"]) // 1000 + 35)

    if premium:
        qs["premium"] = "true"

    extra_headers = _domain_extra_headers(url)
    full_url = f"{SCRAPERAPI_ENDPOINT}?{urlencode(qs)}&url={encoded_url}"

    try:
        return requests.get(full_url, headers=extra_headers, timeout=timeout)
    except Exception:
        return None


def _scrape_via_scraperapi(url: str, api_key: str) -> dict:
    debug = ""
    attempts = _get_attempt_order(url)
    url_lower = url.lower()
    is_nearfinderus = "nearfinderus.com" in url_lower

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

        # nearfinderus extra check: description section must be present
        if is_nearfinderus and attempt["render"] and not _nearfinderus_has_description(text):
            debug += f"  -> nearfinderus: description section not rendered yet ({len(text):,} chars)\n"
            time.sleep(2)
            continue

        return {"success": True, "html": html, "text": text,
                "title": title, "debug": debug + "  -> ScraperAPI OK"}

    return {"success": False, "debug": debug}


# ── Layer 2: Playwright (Windows-safe via subprocess) ────────────────────────

def _scrape_via_playwright(url: str, timeout_ms: int = 45000) -> dict:
    import subprocess
    import json as _json
    import sys
    import os

    url_lower = url.lower()
    for domain, cfg in JS_RENDER_CONFIG.items():
        if domain in url_lower:
            timeout_ms = max(timeout_ms, cfg.get("playwright_timeout", int(cfg["wait"]) + 15000))
            break

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "playwright_worker.py")
    if not os.path.exists(worker):
        return {"success": False, "debug": f"playwright_worker.py not found at {worker}"}

    try:
        proc = subprocess.run(
            [sys.executable, worker, url, str(timeout_ms)],
            capture_output=True,
            text=True,
            timeout=130,
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout.strip()
        if not stdout:
            stderr_snippet = proc.stderr.strip()[:300] if proc.stderr else "no output"
            return {"success": False,
                    "debug": f"Playwright worker produced no output. stderr: {stderr_snippet}"}
        result = _json.loads(stdout)
        if result.get("success") and _is_redirected(url, result.get("text", "")):
            result["success"] = False
            result["debug"]   = (result.get("debug", "") +
                                  "\n  -> Playwright: silently redirected to home/search page")
        return result
    except subprocess.TimeoutExpired:
        return {"success": False, "debug": "Playwright worker timed out (130s)"}
    except _json.JSONDecodeError as e:
        return {"success": False, "debug": f"Playwright worker bad JSON: {e}"}
    except Exception as e:
        return {"success": False, "debug": f"Playwright subprocess error: {e}"}


# ── Combined scraper ──────────────────────────────────────────────────────────

def scrape_page(url: str, api_key: str) -> dict:
    result = {"url": url, "html": "", "text": "", "title": "",
              "error": None, "_debug": ""}

    sa = _scrape_via_scraperapi(url, api_key)
    result["_debug"] += "[ScraperAPI]\n" + sa.get("debug", "") + "\n"

    if sa["success"]:
        result.update({"html": sa["html"], "text": sa["text"], "title": sa["title"]})
        return result

    result["_debug"] += "\n[Playwright fallback]\n"
    pw = _scrape_via_playwright(url)
    result["_debug"] += pw.get("debug", "") + "\n"

    if pw["success"]:
        result.update({"html": pw["html"], "text": pw["text"], "title": pw["title"]})
        return result

    result["error"] = (
        "Both ScraperAPI and Playwright failed to retrieve usable content. "
        "See debug info for details."
    )
    return result


def scrape_batch(urls: list, api_key: str, batch_size: int = 5) -> list:
    sa_results = [None] * len(urls)

    def fetch_sa(index_url):
        idx, url = index_url
        return idx, _scrape_via_scraperapi(url, api_key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(fetch_sa, (i, url)): i
                   for i, url in enumerate(urls)}
        for future in concurrent.futures.as_completed(futures):
            idx, sa = future.result()
            sa_results[idx] = sa

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
    for tag in ["script", "style", "head", "noscript", "iframe"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                      flags=re.DOTALL | re.IGNORECASE)
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

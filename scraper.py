"""
scraper.py
Two-layer scraping strategy:
  Layer 1: ScraperAPI (parallel, handles most sites)
  Layer 2: Playwright fallback via subprocess (Windows/Streamlit safe)

The Playwright layer calls playwright_worker.py as a child process so
it gets its own event loop — no asyncio conflicts with Streamlit/Windows.
"""

import re
import time
import concurrent.futures
import requests
from bs4 import BeautifulSoup


SCRAPERAPI_ENDPOINT = "http://api.scraperapi.com"

JS_RENDER_FIRST_DOMAINS = [
    "nearfinderus.com",
    "us.enrollbusiness.com",
    "band.us",
]

def _get_attempt_order(url: str) -> list:
    """Return scrape attempt order — JS-heavy sites get render=true first."""
    url_lower = url.lower()
    if any(d in url_lower for d in JS_RENDER_FIRST_DOMAINS):
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
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_blocked(html: str, text: str) -> bool:
    combined = (html[:3000] + text[:1000]).lower()
    return any(signal in combined for signal in BLOCK_SIGNALS)


def _is_thin(text: str, min_chars: int = 200) -> bool:
    return len(text.strip()) < min_chars


def _parse(html: str) -> tuple:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg",
                     "header", "footer", "nav"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text  = soup.get_text(separator="\n", strip=True)
    text  = re.sub(r"\n{3,}", "\n\n", text)
    return text, title


# ── Layer 1: ScraperAPI ───────────────────────────────────────────────────────

def _scraperapi_fetch(url, api_key, render, premium=False, timeout=70):
    from urllib.parse import quote, urlencode
    # Manually encode the target URL to preserve special chars like +
    # Using requests params= would corrupt + signs causing HTTP 500
    encoded_url = quote(url, safe="")
    qs = {
        "api_key":      api_key,
        "render":       "true" if render else "false",
        "country_code": "us",
        "keep_headers": "true",
    }
    if render:
        # Wait 8s for JS-heavy pages to lazy-load images and expand hidden sections
        qs["wait_for_selector"] = "body"
        qs["wait"] = "12000"
    if premium:
        qs["premium"] = "true"
    full_url = f"{SCRAPERAPI_ENDPOINT}?{urlencode(qs)}&url={encoded_url}"
    try:
        return requests.get(full_url, timeout=timeout)
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
        if _is_thin(text):
            debug += f"  -> too thin ({len(text.strip())} chars)\n"
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

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playwright_worker.py")
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
            timeout=120,   # hard wall-clock limit
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
        return _json.loads(stdout)
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
        result.update({"html": sa["html"], "text": sa["text"], "title": sa["title"]})
        return result

    # Layer 2: Playwright
    result["_debug"] += "\n[Playwright fallback]\n"
    pw = _scrape_via_playwright(url)
    result["_debug"] += pw.get("debug", "") + "\n"

    if pw["success"]:
        result.update({"html": pw["html"], "text": pw["text"], "title": pw["title"]})
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
"""
playwright_worker.py
Standalone script called by scraper.py via subprocess to avoid
Windows asyncio/Streamlit event loop conflicts.

Usage: python playwright_worker.py <url> <timeout_ms>
Output: JSON to stdout  {"success": bool, "html": "...", "text": "...", "title": "...", "debug": "..."}
"""

import sys
import json
import asyncio
import re

def set_windows_event_loop():
    """Force SelectorEventLoop on Windows — required for subprocess_exec."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BLOCK_SIGNALS = [
    "captcha", "are you human", "cf-browser-verification",
    "ddos-guard", "checking your browser", "verify you are human",
    "enable cookies to continue", "please enable cookies",
    "security check", "access to this page has been denied",
]

def _is_blocked(html, text):
    combined = (html[:3000] + text[:1000]).lower()
    return any(s in combined for s in BLOCK_SIGNALS)

def _is_thin(text, min_chars=200):
    return len(text.strip()) < min_chars

async def scrape(url, timeout):
    from playwright.async_api import async_playwright
    result = {"success": False, "html": "", "text": "", "title": "", "debug": ""}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
            )
            page = await context.new_page()
            try:
                await page.goto(url, timeout=timeout, wait_until="networkidle")
            except Exception:
                try:
                    await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                except Exception as e:
                    result["debug"] = f"goto failed: {e}"
                    await browser.close()
                    return result

            # ── Dismiss cookie/consent banners before scrolling ────────────
            # Many directories (e.g. askmap.net) block image loading behind a
            # cookie consent banner. Click the accept button if one is present.
            await page.evaluate("""() => {
                const acceptTexts = [
                    'i agree', 'accept', 'accept all', 'allow all',
                    'got it', 'ok', 'okay', 'agree', 'accept cookies',
                    'allow cookies', 'consent', 'close',
                ];
                const els = document.querySelectorAll(
                    'a, button, input[type="button"], input[type="submit"]'
                );
                for (const el of els) {
                    const txt = (el.innerText || el.value || '').toLowerCase().trim();
                    if (acceptTexts.includes(txt)) {
                        try { el.click(); } catch(e) {}
                        break;
                    }
                }
            }""")
            await page.wait_for_timeout(1500)

            # ── Scroll entire page to trigger lazy-loaded images and content ──
            await page.wait_for_timeout(2000)
            await page.evaluate("""async () => {
                await new Promise(resolve => {
                    let total = document.body.scrollHeight;
                    let current = 0;
                    let step = 400;
                    const timer = setInterval(() => {
                        window.scrollBy(0, step);
                        current += step;
                        if (current >= total) {
                            clearInterval(timer);
                            window.scrollTo(0, 0);
                            resolve();
                        }
                    }, 120);
                });
            }""")
            await page.wait_for_timeout(2000)

            # ── Expand all collapsed/hidden text sections ──────────────────
            # This handles "See More", "Show more", max-height collapsing, etc.
            await page.evaluate("""() => {
                // Force-show all hidden elements that contain text
                document.querySelectorAll('*').forEach(el => {
                    const style = window.getComputedStyle(el);
                    const isHidden = (
                        style.display === 'none' ||
                        style.visibility === 'hidden' ||
                        style.opacity === '0' ||
                        (style.maxHeight && style.maxHeight !== 'none' && parseInt(style.maxHeight) < 50 && el.innerText && el.innerText.trim().length > 20)
                    );
                    if (isHidden && el.innerText && el.innerText.trim().length > 10) {
                        el.style.display = 'block';
                        el.style.visibility = 'visible';
                        el.style.opacity = '1';
                        el.style.maxHeight = 'none';
                        el.style.overflow = 'visible';
                    }
                });
                // Also click any "See More" / "Show more" buttons
                document.querySelectorAll('a, button, span').forEach(el => {
                    const txt = (el.innerText || '').toLowerCase().trim();
                    if (txt === 'see more' || txt === 'show more' || txt === 'read more' || txt === 'ver más') {
                        try { el.click(); } catch(e) {}
                    }
                });
            }""")
            await page.wait_for_timeout(1500)

            html  = await page.content()
            title = await page.title()

            # ── Extract text WITHOUT removing hidden elements ───────────────
            # We already expanded them above; removing display:none now would
            # strip content that was just made visible by our JS above.
            text  = await page.evaluate("""() => {
                const els = document.querySelectorAll(
                    'script,style,noscript,iframe,svg'
                );
                els.forEach(el => el.remove());
                return document.body ? document.body.innerText : '';
            }""")

            await browser.close()

            if _is_blocked(html, text):
                result["debug"] = "Playwright: blocked/CAPTCHA"
                return result
            if _is_thin(text):
                result["debug"] = f"Playwright: too thin ({len(text.strip())} chars)"
                return result

            result.update({
                "success": True, "html": html, "text": text,
                "title": title,
                "debug": f"Playwright OK | text={len(text):,} chars",
            })
    except Exception as e:
        result["debug"] = f"Playwright exception: {e}"
    return result


if __name__ == "__main__":
    set_windows_event_loop()
    url     = sys.argv[1] if len(sys.argv) > 1 else ""
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 45000
    loop    = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(scrape(url, timeout))
    except Exception as e:
        result = {"success": False, "html": "", "text": "", "title": "",
                  "debug": f"worker top-level error: {e}"}
    finally:
        loop.close()
    # Write JSON to stdout — scraper.py reads this
    print(json.dumps(result, ensure_ascii=False))

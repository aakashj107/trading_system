"""
tv_scraper.py — fetch Pine source from a PUBLIC TradingView script URL.

TradingView serves the source in the page for open/public scripts. We try a
fast requests+regex extraction first; if it fails we fall back to a headless
browser via Playwright/Selenium (only if the user installed it).

PRIVACY: only works for PUBLIC scripts (those without a login wall). Private
or invite-only scripts will not yield source — paste the code manually then.
"""

import re
import sys
import os

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def _extract_from_html(html: str) -> str:
    """Find the Pine source inside a TradingView script page HTML."""
    # TV embeds the source in a JS string; the Give feedback pattern varies.
    # Most reliable: look for 'study(' or 'indicator(' or 'strategy(' and grab
    # a balanced-ish chunk. We look for the largest script-like block.
    markers = ["//@version=", "study(", "indicator(", "strategy("]
    best = ""
    for m in re.finditer(r"(?://@version=\d+[\s\S]{0,20000})", html):
        chunk = m.group(0)
        if len(chunk) > len(best):
            best = chunk
    for mk in markers:
        if mk in html:
            start = html.find(mk)
            chunk = html[start: start + 20000]
            # trim at a clear end (closing of the last function)
            if len(chunk) > len(best):
                best = chunk
    # crude balance trim: cut after last '}' that closes 'study/strategy'
    return best


def fetch_pine_source(url: str, timeout: int = 20) -> str:
    """Return Pine source string, or raise RuntimeError with a clear message."""
    if "tradingview.com" not in url:
        raise ValueError("URL must be a tradingview.com script link")

    # 1) fast path: requests
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        if r.status_code == 200:
            src = _extract_from_html(r.text)
            if len(src) > 80:
                return _clean_scraped(src)
    except Exception:
        pass

    # 2) fallback: headless browser
    for driver_name in ("playwright", "selenium"):
        try:
            if driver_name == "playwright":
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    b = p.chromium.launch(headless=True)
                    pg = b.new_page()
                    pg.goto(url, timeout=timeout * 1000)
                    html = pg.content()
                    b.close()
                src = _extract_from_html(html)
                if len(src) > 80:
                    return _clean_scraped(src)
            else:
                from selenium import webdriver
                opts = webdriver.ChromeOptions()
                opts.add_argument("--headless=new")
                d = webdriver.Chrome(options=opts)
                d.get(url)
                html = d.page_source
                d.quit()
                src = _extract_from_html(html)
                if len(src) > 80:
                    return _clean_scraped(src)
        except Exception:
            continue

    raise RuntimeError(
        "Could not auto-fetch source. The script may be private/invite-only, "
        "or TradingView changed its markup. Paste the Pine code manually instead."
    )


def _clean_scraped(src: str) -> str:
    """Best-effort trim of scraped source down to the Pine block."""
    # cut at the first obvious HTML boundary if present
    for boundary in ("</script>", "</div>", "var collectedScripts"):
        if boundary in src:
            src = src.split(boundary)[0]
    return src.strip()

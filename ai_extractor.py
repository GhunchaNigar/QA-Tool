"""
ai_extractor.py
Sends scraped page content to Google Gemini and extracts
business fields as JSON — including visual fields (Logo, Photos).
No hardcoded layout assumptions. Gemini searches the whole page.

v2 — Robust logo/photo detection with per-source HTML pre-extraction.
"""

import json
import re
from google import genai
from google.genai import types
from fields_config import VISUAL_FIELDS


GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Image-quality helpers
# ─────────────────────────────────────────────────────────────────────────────

# Src patterns that reliably indicate site-UI / non-business images
_UI_IMAGE_PATTERNS = re.compile(
    r"(icon|sprite|arrow|chevron|star-rating|rating-star|badge|flag|"
    r"banner-ad|advertisement|ad[-_]|[-_]ad\.|pixel\.gif|blank\.gif|"
    r"spacer|placeholder|loading|spinner|ajax-loader|no[-_]?image|"
    r"default[-_]avatar|generic[-_])",
    re.IGNORECASE,
)

# Src patterns that reliably indicate a logo
_LOGO_SRC_PATTERNS = re.compile(
    r"(logo|logos|thumb_|profile[-_]?img|profile[-_]?pic|avatar|brand|"
    r"business[-_]?img|company[-_]?img|listing[-_]?img|profile[-_]?image|"
    r"BusinessProfile|biz[-_]?logo)",
    re.IGNORECASE,
)

# Src patterns that reliably indicate a photo / gallery image
_PHOTO_SRC_PATTERNS = re.compile(
    r"(photo|photos|gallery|image|images|media|cover|banner|hero|"
    r"carousel|slide|slider|backdrop|background|uploads|BusinessPhoto|"
    r"biz[-_]?photo|listing[-_]?photo)",
    re.IGNORECASE,
)

# CSS class / id patterns for logo containers
_LOGO_CLASS_PATTERNS = re.compile(
    r"(logo|brand|business[-_]?thumb|profile[-_]?img|listing[-_]?logo|"
    r"company[-_]?logo|biz[-_]?logo|business[-_]?logo|profile[-_]?logo|"
    r"thumb[-_]?wrap|logo[-_]?wrap|profile[-_]?pic|avatar)",
    re.IGNORECASE,
)

# CSS class / id patterns for photo / gallery containers
_PHOTO_CLASS_PATTERNS = re.compile(
    r"(gallery|carousel|slider|slideshow|photo[-_]?section|banner|hero|"
    r"cover[-_]?photo|cover[-_]?image|featured[-_]?image|listing[-_]?image|"
    r"business[-_]?photo|profile[-_]?banner|backdrop|media[-_]?section)",
    re.IGNORECASE,
)


def _all_srcs(tag) -> list:
    """
    Return all non-data-URI image URLs from a tag, checking every
    lazy-load attribute variant used in the wild.
    """
    attrs = ("src", "data-src", "data-lazy", "data-lazy-src",
             "data-original", "data-url", "data-image",
             "data-bg", "data-background", "data-srcset")
    result = []
    for attr in attrs:
        v = tag.get(attr, "").strip()
        if v and not v.startswith("data:") and v not in result:
            result.append(v)
    # Also pull first URL from srcset
    srcset = tag.get("srcset", "").strip()
    if srcset:
        first = srcset.split(",")[0].strip().split()[0]
        if first and not first.startswith("data:") and first not in result:
            result.append(first)
    return result


def _img_dimensions(img) -> tuple:
    """Return (width, height) as ints. Returns (0,0) if unreadable."""
    try:
        w = int(img.get("width", 0) or 0)
        h = int(img.get("height", 0) or 0)
        return w, h
    except (ValueError, TypeError):
        return 0, 0


def _is_tiny(img, threshold: int = 50) -> bool:
    """True if the image has explicit dimensions smaller than threshold."""
    w, h = _img_dimensions(img)
    if w > 0 and w < threshold:
        return True
    if h > 0 and h < threshold:
        return True
    return False


def _is_ui_image(src: str) -> bool:
    """True if src looks like a site-UI / non-business image."""
    return bool(_UI_IMAGE_PATTERNS.search(src))


def _cls_str(tag) -> str:
    """Return joined class list + id as a single lowercase string."""
    classes = " ".join(tag.get("class", []))
    id_val = tag.get("id", "")
    return (classes + " " + id_val).lower()


def _ancestor_cls(tag, depth: int = 4) -> str:
    """Return combined class+id string of up to `depth` ancestors."""
    parts = []
    current = tag.parent
    for _ in range(depth):
        if current is None or not hasattr(current, "get"):
            break
        parts.append(_cls_str(current))
        current = current.parent
    return " ".join(parts)


def _css_background_images(soup) -> list:
    """
    Extract URLs from inline style="background-image: url(...)" attributes.
    These are commonly used for hero banners on business directories.
    """
    urls = []
    for tag in soup.find_all(style=True):
        style_val = tag.get("style", "")
        found = re.findall(
            r"background(?:-image)?\s*:\s*url\(['\"]?([^'\"\)]+)['\"]?\)",
            style_val, re.IGNORECASE,
        )
        for url in found:
            url = url.strip()
            if url and not url.startswith("data:") and url not in urls:
                urls.append(url)
    return urls


def _is_hidden_tag(tag) -> bool:
    """Return True if the tag or any ancestor has display:none / visibility:hidden."""
    for el in [tag] + list(tag.parents):
        if not hasattr(el, "get"):
            continue
        style = el.get("style", "").lower()
        if "display: none" in style or "display:none" in style:
            return True
        if "visibility: hidden" in style or "visibility:hidden" in style:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Logo detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_logo(soup, source: str = "") -> str:
    """
    Multi-strategy logo detection.
    Returns the img tag HTML string, or "" if not found.
    """
    source_lower = source.lower()

    # ── Strategy 1: src attribute matches logo patterns ───────────────────
    for img in soup.find_all("img"):
        for src in _all_srcs(img):
            if _LOGO_SRC_PATTERNS.search(src) and not _is_tiny(img, 20):
                return str(img)

    # ── Strategy 2: img class / id / alt contains logo signals ───────────
    for img in soup.find_all("img"):
        img_cls = _cls_str(img)
        alt = img.get("alt", "").lower()
        if (re.search(r"(logo|brand|emblem|crest)", img_cls) or
                re.search(r"(logo|brand)", alt)):
            srcs = _all_srcs(img)
            if srcs and not _is_tiny(img, 20):
                return str(img)

    # ── Strategy 3: parent/ancestor container has logo class ─────────────
    for img in soup.find_all("img"):
        anc = _ancestor_cls(img, depth=5)
        if _LOGO_CLASS_PATTERNS.search(anc):
            srcs = _all_srcs(img)
            if srcs and not _is_tiny(img, 20) and not _is_ui_image(srcs[0]):
                return str(img)

    # ── Strategy 4: enrollbusiness / similar — circular overlay logo ──────
    # These sites overlay a circular logo on top of the hero banner.
    # The logo img is typically inside a div with class containing
    # 'profile', 'thumb', 'overlay', 'logo', 'circle', or 'round'.
    overlay_signals = re.compile(
        r"(profile|thumb|overlay|circle|round|badge|seal|"
        r"business[-_]?icon|company[-_]?icon|listing[-_]?icon)",
        re.IGNORECASE,
    )
    for container in soup.find_all(["div", "figure", "span", "a"]):
        if overlay_signals.search(_cls_str(container)):
            img = container.find("img")
            if img:
                srcs = _all_srcs(img)
                if srcs and not _is_tiny(img, 30) and not _is_ui_image(srcs[0]):
                    return str(img)

    # ── Strategy 5: enrollbusiness-specific — first img in the hero/cover ─
    if "enrollbusiness" in source_lower:
        hero_signals = re.compile(
            r"(hero|banner|cover|carousel|slider|featured|"
            r"header[-_]?img|top[-_]?img|main[-_]?img)",
            re.IGNORECASE,
        )
        for container in soup.find_all(["div", "section", "header", "figure"]):
            if hero_signals.search(_cls_str(container)):
                imgs_in_hero = container.find_all("img")
                for img in imgs_in_hero:
                    srcs = _all_srcs(img)
                    if srcs and not _is_tiny(img, 30) and not _is_ui_image(srcs[0]):
                        # The SECOND image inside a hero is usually the circular overlay logo
                        # (first is the background photo, second is the logo overlay)
                        return str(img)

    # ── Strategy 6: meta og:image as last resort ──────────────────────────
    og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    if og and og.get("content", "").startswith("http"):
        return f'<img src="{og["content"]}" data-source="og:image">'

    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Photo detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_photos(soup, source: str = "") -> bool:
    """
    Multi-strategy photo detection. Returns True if business photos exist.

    Checks: gallery containers, carousel/slider, hero banners,
    CSS background-images, photo tabs, og:image, and general large images.
    """
    source_lower = source.lower()

    # ── Strategy 1: gallery / photo section containers ────────────────────
    for container in soup.find_all(["div", "section", "ul", "figure"]):
        cls = _cls_str(container)
        if _PHOTO_CLASS_PATTERNS.search(cls):
            imgs = container.find_all("img")
            for img in imgs:
                srcs = _all_srcs(img)
                if srcs and not _is_tiny(img, 40) and not _is_ui_image(srcs[0]):
                    return True
            # Even if no <img> found, check CSS background in this container
            bg_urls = _css_background_images(
                type("_", (), {"find_all": lambda self, **kw: [container]})()
            )
            # Simpler: just check the container's own style
            style = container.get("style", "")
            if "background" in style.lower() and "url(" in style.lower():
                return True

    # ── Strategy 2: CSS background-image on any element ───────────────────
    bg_urls = _css_background_images(soup)
    for url in bg_urls:
        if not _is_ui_image(url):
            return True

    # ── Strategy 3: enrollbusiness / sites with hero banner ──────────────
    # Hero/banner image = business photo. If the page has a prominent
    # top-of-page image (carousel, slider, or cover), that counts.
    hero_signals = re.compile(
        r"(hero|banner|cover|carousel|slider|slideshow|"
        r"featured|backdrop|jumbotron|masthead)",
        re.IGNORECASE,
    )
    for container in soup.find_all(["div", "section", "header", "figure"]):
        if hero_signals.search(_cls_str(container)):
            imgs = container.find_all("img")
            for img in imgs:
                srcs = _all_srcs(img)
                if srcs and not _is_tiny(img, 40) and not _is_ui_image(srcs[0]):
                    return True

    # ── Strategy 4: photo src patterns ───────────────────────────────────
    for img in soup.find_all("img"):
        for src in _all_srcs(img):
            if _PHOTO_SRC_PATTERNS.search(src) and not _is_ui_image(src):
                if not _is_tiny(img, 40):
                    return True

    # ── Strategy 5: anchor / tab linking to a Photos page ─────────────────
    # Enrollbusiness has a "Photos" tab — even if the photos aren't loaded
    # on the current page, the tab being present confirms photos exist.
    for a in soup.find_all("a"):
        txt = a.get_text(strip=True).lower()
        href = a.get("href", "").lower()
        if txt in ("photos", "photo", "gallery", "images") or "photo" in href:
            # Make sure it's not just a nav link to another site
            if not any(d in href for d in ("google", "facebook", "twitter", "instagram")):
                return True

    # ── Strategy 6: og:image ──────────────────────────────────────────────
    og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    if og and og.get("content", "").startswith("http"):
        og_url = og["content"]
        if not _is_ui_image(og_url):
            return True

    # ── Strategy 7: any sufficiently large, non-UI <img> ─────────────────
    # This is a liberal fallback: if a listing page has any real image at all,
    # it almost certainly has a business photo.
    large_img_count = 0
    for img in soup.find_all("img"):
        srcs = _all_srcs(img)
        if not srcs:
            continue
        src = srcs[0]
        if _is_ui_image(src):
            continue
        if _is_tiny(img, 60):
            continue
        # Skip common site-framework images
        if any(x in src.lower() for x in ("favicon", "sprite", "icon-", "-icon")):
            continue
        large_img_count += 1
        if large_img_count >= 2:
            # Two or more real images on the page = photos are present
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Main pre-extraction function (replaces old _extract_page_hints)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_hints(page_html: str, page_text: str, source: str = "") -> dict:
    """
    Structurally extract all fields from page HTML using BeautifulSoup.
    Returns a hints dict with confirmed values that are passed to the
    AI prompt as authoritative — no re-interpretation needed.
    """
    hints = {
        "logo_html": "",
        "logo_confirmed": False,
        "photos_confirmed": False,
        "description_text": "",
        "website_url": "",
        "social_links": "",
        "gbp_link": "",
    }
    if not page_html:
        return hints

    try:
        from bs4 import BeautifulSoup
        from urllib.parse import unquote
        soup = BeautifulSoup(page_html, "html.parser")
        source_lower = source.lower()

        # ── LOGO ─────────────────────────────────────────────────────────
        logo_html = _detect_logo(soup, source)
        if logo_html:
            hints["logo_html"] = logo_html
            hints["logo_confirmed"] = True

        # ── PHOTOS ───────────────────────────────────────────────────────
        hints["photos_confirmed"] = _detect_photos(soup, source)

        # ── DESCRIPTION ───────────────────────────────────────────────────

        def _good_desc(text: str) -> bool:
            if len(text) < 40:
                return False
            bad_phrases = (
                "payment methods", "last update", "general information",
                "write a review", "sign up", "site map", "privacy policy",
                "terms of", "cookie", "copyright", "all rights reserved",
                "related companies", "opening hours", "phone number",
                "categories", "social", "directions", "get directions",
                "claim this", "report an error", "edit this",
            )
            tl = text.lower()
            return not any(b in tl for b in bad_phrases)

        def _is_hidden(tag) -> bool:
            return _is_hidden_tag(tag)

        # enrollbusiness-specific description extraction
        if "enrollbusiness" in source_lower:
            # P1: look for section/div labeled "About" or "Description"
            for tag in soup.find_all(["div", "section", "article"]):
                cls = _cls_str(tag)
                if re.search(r"(about|description|overview|summary|info[-_]?text)", cls):
                    text = tag.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        hints["description_text"] = text
                        break
            # P2: heading "About" / "Description" → next sibling
            if not hints["description_text"]:
                for h in soup.find_all(["h1","h2","h3","h4","h5","strong"]):
                    htxt = h.get_text(strip=True).lower()
                    if htxt in ("about", "description", "about us", "overview"):
                        for sib in h.find_next_siblings(["p", "div"]):
                            text = sib.get_text(separator=" ", strip=True)
                            if _good_desc(text):
                                hints["description_text"] = text
                                break
                        if hints["description_text"]:
                            break
            # P3: longest visible paragraph
            if not hints["description_text"]:
                best = ""
                for p in soup.find_all("p"):
                    if _is_hidden(p):
                        continue
                    text = p.get_text(separator=" ", strip=True)
                    if _good_desc(text) and len(text) > len(best):
                        best = text
                hints["description_text"] = best

        elif "nearfinderus" in source_lower:
            # P1: div.mt-4 > visible p
            for div in soup.find_all("div"):
                if "mt-4" in " ".join(div.get("class", [])):
                    for p in div.find_all("p"):
                        if not _is_hidden(p):
                            text = p.get_text(separator=" ", strip=True)
                            if _good_desc(text):
                                hints["description_text"] = text
                                break
                if hints["description_text"]:
                    break
            # P2: heading "More about" → next sibling
            if not hints["description_text"]:
                for h in soup.find_all(["h2","h3","h4","strong","b"]):
                    if "more about" in h.get_text().lower():
                        for sib in h.find_next_siblings(["p","div"]):
                            text = sib.get_text(separator=" ", strip=True)
                            if _good_desc(text):
                                hints["description_text"] = text
                                break
                        if hints["description_text"]:
                            break
            # P3: longest visible p
            if not hints["description_text"]:
                best = ""
                for p in soup.find_all("p"):
                    if _is_hidden(p):
                        continue
                    text = p.get_text(separator=" ", strip=True)
                    if _good_desc(text) and len(text) > len(best):
                        best = text
                hints["description_text"] = best

        elif "askmap" in source_lower:
            for tag in soup.find_all(["div","p","section"]):
                cls = _cls_str(tag)
                if any(x in cls for x in ("description","about","info","detail","content")):
                    text = tag.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        hints["description_text"] = text
                        break
            if not hints["description_text"]:
                best = ""
                for p in soup.find_all("p"):
                    if _is_hidden(p):
                        continue
                    text = p.get_text(separator=" ", strip=True)
                    if _good_desc(text) and len(text) > len(best):
                        best = text
                hints["description_text"] = best

        else:
            # Generic: longest visible paragraph
            best = ""
            for p in soup.find_all("p"):
                if _is_hidden(p):
                    continue
                text = p.get_text(separator=" ", strip=True)
                if _good_desc(text) and len(text) > len(best):
                    best = text
            hints["description_text"] = best

        # ── WEBSITE URL ───────────────────────────────────────────────────
        skip_domains = (
            "enrollbusiness.com", "nearfinderus.com", "hotfrog.com",
            "brownbook.net", "freelistingusa.com", "smallbusinessusa.com",
            "askmap.net", "google.com", "facebook.com", "instagram.com",
            "twitter.com", "linkedin.com", "whatsapp.com", "youtube.com",
            "yelp.com",
        )

        if "nearfinderus" in source_lower:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/empresa/redirect?url=" in href:
                    raw = href.split("url=")[1].split("&")[0]
                    decoded = unquote(raw)
                    if not any(s in decoded for s in ("whatsapp.com", "nearfinderus.com")):
                        hints["website_url"] = decoded
                        break
            if not hints["website_url"]:
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("http") and not any(s in href for s in skip_domains):
                        hints["website_url"] = href
                        break

        elif "enrollbusiness" in source_lower:
            # enrollbusiness often has a "Visit Website" link or an external URL field
            website_labels = re.compile(
                r"(visit website|website|web site|official site|homepage)",
                re.IGNORECASE,
            )
            for a in soup.find_all("a", href=True):
                href = a["href"]
                txt = a.get_text(strip=True)
                if website_labels.search(txt) and href.startswith("http"):
                    if not any(s in href for s in skip_domains):
                        hints["website_url"] = href
                        break
            # Fallback: any outbound link not in skip list
            if not hints["website_url"]:
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("http") and not any(s in href for s in skip_domains):
                        hints["website_url"] = href
                        break

        # ── SOCIAL MEDIA LINKS ────────────────────────────────────────────
        social_domains = [
            "facebook.com", "instagram.com", "linkedin.com",
            "twitter.com", "x.com", "youtube.com", "tiktok.com",
            "whatsapp.com", "pinterest.com",
        ]
        social_links = []

        if "nearfinderus" in source_lower:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/empresa/redirect?url=" in href:
                    raw = href.split("url=")[1].split("&")[0]
                    decoded = unquote(raw)
                    if any(s in decoded for s in social_domains):
                        social_links.append(decoded)
        else:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if any(s in href for s in social_domains):
                    social_links.append(href)

        # Deduplicate while preserving order
        if social_links:
            hints["social_links"] = ", ".join(dict.fromkeys(social_links))

        # ── GBP LINK ──────────────────────────────────────────────────────
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "google.com" in href and any(p in href for p in ("maps/place","maps?q","goo.gl")):
                hints["gbp_link"] = href
                break

    except Exception:
        pass

    return hints


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(page_text: str, page_html: str, fields: list, source: str = "") -> str:
    """
    Build a generic extraction prompt.
    HTML is always included so the model can detect images for visual fields.
    Pre-extracted hints are passed as AUTHORITATIVE confirmed values.
    """
    # Pre-extract structured values from HTML using BeautifulSoup
    hints = _extract_page_hints(page_html, page_text, source)

    # Extract ALL img tags from full HTML — for AI's own logo/photo check
    all_img_tags = re.findall(r'<img[^>]*>', page_html, flags=re.IGNORECASE)
    img_section = "\n".join(all_img_tags[:100]) if all_img_tags else ""

    # Extract CSS background-image URLs for the AI
    bg_urls = re.findall(
        r"background(?:-image)?\s*:\s*url\(['\"]?([^'\"\)]+)['\"]?\)",
        page_html, re.IGNORECASE,
    )
    bg_section = "\n".join(bg_urls[:20]) if bg_urls else ""

    # HTML snippet: first 15k + last 5k
    if len(page_html) > 20000:
        html_snippet = page_html[:15000] + "\n\n…[middle omitted]…\n\n" + page_html[-5000:]
    else:
        html_snippet = page_html

    # Build pre-extracted facts section
    pre_extracted_facts = []

    # Visual fields — these are CONFIRMED by our robust BeautifulSoup logic
    if hints["logo_confirmed"]:
        pre_extracted_facts.append(
            f'LOGO CONFIRMED PRESENT (detected via HTML analysis — return "PRESENT" for Logo):\n'
            f'{hints["logo_html"]}'
        )
    if hints["photos_confirmed"]:
        pre_extracted_facts.append(
            'PHOTOS CONFIRMED PRESENT (detected via HTML analysis — return "PRESENT" for Photos)'
        )

    if hints["description_text"]:
        pre_extracted_facts.append(
            f'DESCRIPTION TEXT (confirmed from page HTML — use verbatim for Description field):\n'
            f'{hints["description_text"]}'
        )
    if hints["website_url"]:
        pre_extracted_facts.append(
            f'WEBSITE URL (confirmed from page HTML — use as-is for Website URL field):\n'
            f'{hints["website_url"]}'
        )
    if hints["social_links"]:
        pre_extracted_facts.append(
            f'SOCIAL MEDIA LINKS (confirmed from page HTML):\n{hints["social_links"]}'
        )
    if hints["gbp_link"]:
        pre_extracted_facts.append(
            f'GBP LINK (confirmed from page HTML):\n{hints["gbp_link"]}'
        )

    # Build content section
    parts = []
    if pre_extracted_facts:
        parts.append(
            "═══ PRE-EXTRACTED FIELDS (AUTHORITATIVE — use these directly, do NOT override) ═══\n\n"
            + "\n\n".join(pre_extracted_facts)
        )
    if page_text and len(page_text.strip()) > 100:
        parts.append(f"PAGE TEXT:\n{page_text[:30000]}")
    parts.append(f"PAGE HTML SNIPPET:\n{html_snippet}")
    if img_section:
        parts.append(f"ALL <img> TAGS FROM PAGE (for logo/photo detection):\n{img_section}")
    if bg_section:
        parts.append(
            f"CSS BACKGROUND-IMAGE URLs (these are real images on the page — "
            f"if any look like business photos or a logo, treat them as PRESENT):\n{bg_section}"
        )

    content_section = "\n\n".join(parts)

    # ── Field rules ───────────────────────────────────────────────────────
    field_rules = []
    for f in fields:
        if f == "Name":
            field_rules.append('- "Name": the primary business name on the page.')
        elif f == "Phone":
            field_rules.append(
                '- "Phone": any phone number. Return digits and separators only, '
                'e.g. +14155552671 or 336-517-8789.'
            )
        elif f == "Website URL":
            field_rules.append(
                '- "Website URL": the business\'s own website URL.\n'
                '  If WEBSITE URL appears in PRE-EXTRACTED FIELDS above, use it directly.\n'
                '  Otherwise find a "Visit Website" or external link — NOT the directory domain.'
            )
        elif f == "Street":
            field_rules.append('- "Street": street address (number + street name).')
        elif f == "City":
            field_rules.append('- "City": city name.')
        elif f == "State":
            field_rules.append('- "State": state / region as shown (abbreviation or full name).')
        elif f == "Zipcode":
            field_rules.append('- "Zipcode": postal/zip code.')
        elif f == "Country":
            field_rules.append(
                '- "Country": country name or code (e.g. "US", "United States", "UK"). '
                'Look for flag icons (🇺🇸 = US), "USA" in addresses, or country fields. '
                'A two-letter ISO code is a valid answer.'
            )
        elif f == "Category":
            field_rules.append('- "Category": business type / industry category shown.')
        elif f == "Keywords":
            field_rules.append(
                '- "Keywords": tags, keywords, or labels for the business. '
                'May appear pipe-separated or comma-separated. Return comma-separated.'
            )
        elif f == "Description":
            field_rules.append(
                '- "Description": the business\'s own descriptive text.\n'
                '  If DESCRIPTION TEXT appears in PRE-EXTRACTED FIELDS above, use it verbatim.\n'
                '  Otherwise find prose paragraphs describing what the business does.\n'
                '  EXCLUDE: navigation text, review prompts, "Payment methods", '
                '"Last update", "General information", footer text.\n'
                '  Return null only if no genuine description exists.'
            )
        elif f == "Hours":
            field_rules.append(
                '- "Hours": operating hours, e.g. "Mon-Fri 9am-5pm" or '
                '"Monday - Sunday: 24 Hours Open". Look in Working Hours sections.'
            )
        elif f == "Social Media Links":
            field_rules.append(
                '- "Social Media Links": Facebook, LinkedIn, Twitter/X, Instagram, '
                'WhatsApp, YouTube, TikTok URLs.\n'
                '  If SOCIAL MEDIA LINKS appears in PRE-EXTRACTED FIELDS, use it.\n'
                '  Otherwise search the page. Return comma-separated.'
            )
        elif f == "GBP Link":
            field_rules.append(
                '- "GBP Link": Google Business Profile / Google Maps link.\n'
                '  If GBP LINK appears in PRE-EXTRACTED FIELDS, use it.\n'
                '  Otherwise look for google.com/maps links.'
            )
        elif f == "Business Email":
            field_rules.append('- "Business Email": any business email address on the page.')
        elif f == "Logo":
            field_rules.append(
                '- "Logo": does a business logo exist on this page?\n'
                '  ★ If "LOGO CONFIRMED PRESENT" appears in PRE-EXTRACTED FIELDS above → '
                'return "PRESENT" immediately, no further checking needed.\n'
                '  Otherwise check ALL <img> TAGS for:\n'
                '    * src containing: logo, logos, thumb_, profile, avatar, brand, '
                'BusinessProfile, biz-logo, company-img\n'
                '    * class/alt containing: logo, brand, avatar, profile-img\n'
                '    * Any circular/overlay image on top of a hero banner\n'
                '  Also check CSS BACKGROUND-IMAGE URLs for logo-like patterns.\n'
                '  Return "PRESENT" if found, null if truly absent.'
            )
        elif f == "Photos":
            field_rules.append(
                '- "Photos": do business photos exist on this page?\n'
                '  ★ If "PHOTOS CONFIRMED PRESENT" appears in PRE-EXTRACTED FIELDS above → '
                'return "PRESENT" immediately.\n'
                '  Otherwise check for ANY of these:\n'
                '    * Hero/banner image at the top (carousel, slider, cover photo)\n'
                '    * Gallery or photo section with <img> tags\n'
                '    * CSS background-image on a banner/hero div (see CSS BACKGROUND-IMAGE URLs)\n'
                '    * A "Photos" tab / link on the page (means the business has photos)\n'
                '    * Multiple large images anywhere on the listing\n'
                '    * og:image meta tag\n'
                '  Even a single cover/banner image counts → return "PRESENT".\n'
                '  Return null ONLY if absolutely no images of any kind exist on the page.'
            )

    rules_text = "\n".join(field_rules)

    return f"""You are a business data extraction assistant. Extract listing information from the page content below.

Extract ONLY these fields: {fields}

CRITICAL RULES:
- Return ONLY a valid JSON object. No explanation, no markdown, no backticks.
- Search the ENTIRE content — text, HTML, and img tags — for each field.
- PRE-EXTRACTED FIELDS are AUTHORITATIVE: if a field is confirmed there, use that value directly.
- Use null for fields genuinely absent from the page.
- Do NOT guess or invent values.
- For Logo and Photos: if PRE-EXTRACTED FIELDS confirms PRESENT, return "PRESENT" — do not second-guess.

FIELD INSTRUCTIONS:
{rules_text}

Example:
{{"Name":"Acme Corp","Phone":"+14155552671","City":"San Francisco","State":"CA","Country":"US","Logo":"PRESENT","Photos":"PRESENT"}}

{content_section}"""


# ─────────────────────────────────────────────────────────────────────────────
#  Gemini API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> str:
    raw = raw.strip()
    if raw.endswith("}"):
        return raw
    for candidate in [
        raw + 'null}',
        raw + '"}',
        raw + '"}}',
        raw.rsplit(",", 1)[0] + "}",
    ]:
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    return raw


def _call_gemini(client, prompt: str):
    last_err = None
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192,
                ),
            )
            finish = None
            try:
                finish = response.candidates[0].finish_reason
            except Exception:
                pass
            text = response.text.strip()
            if finish and str(finish) in ("FinishReason.MAX_TOKENS", "MAX_TOKENS", "2"):
                text = text + "__TRUNCATED__"
            return text, model_name
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower():
                last_err = e
                continue
            raise
    raise last_err or RuntimeError("All Gemini models failed")


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_fields(
    page_text: str, page_html: str, fields: list, source: str, api_key: str
) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(page_text, page_html, fields, source)

    raw = ""
    model_used = ""
    try:
        raw, model_used = _call_gemini(client, prompt)
        raw = re.sub(r"```json|```", "", raw).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        truncated = "__TRUNCATED__" in raw
        if truncated:
            raw = raw.replace("__TRUNCATED__", "")

        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(raw)
            extracted = json.loads(repaired)
            extracted["_repaired"] = True

        if truncated:
            extracted["_truncated"] = True
        extracted["_model"] = model_used

        # ── POST-PROCESS: override AI if our pre-extraction is authoritative ──
        # This ensures scraper-confirmed visual fields are never lost to AI
        # hallucination or context truncation.
        hints = _extract_page_hints(page_html, page_text, source)
        if hints["logo_confirmed"] and "Logo" in fields:
            extracted["Logo"] = "PRESENT"
        if hints["photos_confirmed"] and "Photos" in fields:
            extracted["Photos"] = "PRESENT"

    except json.JSONDecodeError as e:
        extracted = {"_parse_error": str(e), "_raw": raw[:800], "_model": model_used}
        # Still apply visual field overrides even if JSON parsing failed for other fields
        try:
            hints = _extract_page_hints(page_html, page_text, source)
            if hints["logo_confirmed"] and "Logo" in fields:
                extracted["Logo"] = "PRESENT"
            if hints["photos_confirmed"] and "Photos" in fields:
                extracted["Photos"] = "PRESENT"
        except Exception:
            pass
    except Exception as e:
        extracted = {"_error": str(e), "_raw": raw[:300], "_model": model_used}

    return extracted


def extract_batch(
    scraped_pages: list,
    fields: list,
    source: str,
    api_key: str,
    progress_callback=None,
) -> list:
    from scraper import clean_html, clean_text

    results = []
    for i, page in enumerate(scraped_pages):
        if page.get("error"):
            result = {f: None for f in fields}
            result["_scrape_error"] = page["error"]
        else:
            cleaned_html = clean_html(page.get("html", ""))
            cleaned_text = clean_text(page.get("text", ""))

            title = page.get("title", "").strip()
            if title and title not in cleaned_text[:200]:
                cleaned_text = f"PAGE TITLE: {title}\n\n{cleaned_text}"

            result = extract_fields(
                cleaned_text, cleaned_html, fields, source, api_key
            )

        result["_url"] = page["url"]
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, len(scraped_pages))

    return results

"""
ai_extractor.py
Sends scraped page content to Google Gemini and extracts
business fields as JSON — including visual fields (Logo, Photos).
No hardcoded layout assumptions. Gemini searches the whole page.

v3 — Full per-domain extraction for all 7 supported directories:
     askmap.net · brownbook.net · freelistingusa.com · hotfrog.com
     nearfinderus.com · smallbusinessusa.com · us.enrollbusiness.com

Key improvements over v2:
  - JSON-LD extraction layer (hotfrog, brownbook, freelistingusa all use it)
  - schema.org microdata extraction (askmap, brownbook)
  - Per-domain description / website / hours / social extractors for the
    4 previously unhandled domains
  - askmap website URL now extracted correctly via itemprop="url" / rel=nofollow
  - brownbook: itemprop-based extraction for all core fields
  - freelistingusa: listing-description div + schema.org fallback
  - hotfrog: JSON-LD primary path, DOM fallback
  - smallbusinessusa: business-description / about div paths
  - Hours field added to hints (pre-extracted and authoritative)
  - _detect_logo / _detect_photos: brownbook itemprop="image" strategy added
  - Redirect guard mirrors scraper.py's REDIRECT_SIGNALS
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

_UI_IMAGE_PATTERNS = re.compile(
    r"(icon|sprite|arrow|chevron|star-rating|rating-star|badge|flag|"
    r"banner-ad|advertisement|ad[-_]|[-_]ad\.|pixel\.gif|blank\.gif|"
    r"spacer|placeholder|loading|spinner|ajax-loader|no[-_]?image|"
    r"default[-_]avatar|generic[-_])",
    re.IGNORECASE,
)

_LOGO_SRC_PATTERNS = re.compile(
    r"(logo|logos|thumb_|profile[-_]?img|profile[-_]?pic|avatar|brand|"
    r"business[-_]?img|company[-_]?img|listing[-_]?img|profile[-_]?image|"
    r"BusinessProfile|biz[-_]?logo)",
    re.IGNORECASE,
)

_PHOTO_SRC_PATTERNS = re.compile(
    r"(photo|photos|gallery|image|images|media|cover|banner|hero|"
    r"carousel|slide|slider|backdrop|background|uploads|BusinessPhoto|"
    r"biz[-_]?photo|listing[-_]?photo)",
    re.IGNORECASE,
)

_LOGO_CLASS_PATTERNS = re.compile(
    r"(logo|brand|business[-_]?thumb|profile[-_]?img|listing[-_]?logo|"
    r"company[-_]?logo|biz[-_]?logo|business[-_]?logo|profile[-_]?logo|"
    r"thumb[-_]?wrap|logo[-_]?wrap|profile[-_]?pic|avatar)",
    re.IGNORECASE,
)

_PHOTO_CLASS_PATTERNS = re.compile(
    r"(gallery|carousel|slider|slideshow|photo[-_]?section|banner|hero|"
    r"cover[-_]?photo|cover[-_]?image|featured[-_]?image|listing[-_]?image|"
    r"business[-_]?photo|profile[-_]?banner|backdrop|media[-_]?section)",
    re.IGNORECASE,
)


def _all_srcs(tag) -> list:
    """Return all non-data-URI image URLs from a tag, checking every lazy-load variant."""
    attrs = ("src", "data-src", "data-lazy", "data-lazy-src",
             "data-original", "data-url", "data-image",
             "data-bg", "data-background", "data-srcset")
    result = []
    for attr in attrs:
        v = tag.get(attr, "").strip()
        if v and not v.startswith("data:") and v not in result:
            result.append(v)
    srcset = tag.get("srcset", "").strip()
    if srcset:
        first = srcset.split(",")[0].strip().split()[0]
        if first and not first.startswith("data:") and first not in result:
            result.append(first)
    return result


def _img_dimensions(img) -> tuple:
    try:
        w = int(img.get("width", 0) or 0)
        h = int(img.get("height", 0) or 0)
        return w, h
    except (ValueError, TypeError):
        return 0, 0


def _is_tiny(img, threshold: int = 50) -> bool:
    w, h = _img_dimensions(img)
    if w > 0 and w < threshold:
        return True
    if h > 0 and h < threshold:
        return True
    return False


def _is_ui_image(src: str) -> bool:
    return bool(_UI_IMAGE_PATTERNS.search(src))


def _cls_str(tag) -> str:
    classes = " ".join(tag.get("class", []))
    id_val = tag.get("id", "")
    return (classes + " " + id_val).lower()


def _ancestor_cls(tag, depth: int = 4) -> str:
    parts = []
    current = tag.parent
    for _ in range(depth):
        if current is None or not hasattr(current, "get"):
            break
        parts.append(_cls_str(current))
        current = current.parent
    return " ".join(parts)


def _css_background_images(soup) -> list:
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
#  JSON-LD extractor  (hotfrog, brownbook, freelistingusa all embed it)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_ld(soup) -> list:
    """
    Parse all <script type="application/ld+json"> blocks on the page.
    Returns a list of dicts (may be empty).
    """
    results = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or ""
            raw = raw.strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        except Exception:
            continue
    return results


def _ld_find(ld_blocks: list, *types_) -> dict:
    """
    Return the first JSON-LD block whose @type matches one of types_.
    Types are matched case-insensitively and support partial match.
    """
    type_lower = [t.lower() for t in types_]
    for block in ld_blocks:
        bt = block.get("@type", "")
        if isinstance(bt, list):
            bt = " ".join(bt)
        if any(t in bt.lower() for t in type_lower):
            return block
    return {}


def _ld_address(ld: dict) -> dict:
    """Extract address sub-fields from a JSON-LD LocalBusiness block."""
    addr = ld.get("address", {})
    if isinstance(addr, str):
        return {}
    return {
        "street":  addr.get("streetAddress", ""),
        "city":    addr.get("addressLocality", ""),
        "state":   addr.get("addressRegion", ""),
        "zip":     addr.get("postalCode", ""),
        "country": addr.get("addressCountry", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  schema.org microdata extractor  (askmap, brownbook)
# ─────────────────────────────────────────────────────────────────────────────

def _itemprop(soup, prop: str, attr: str = "text") -> str:
    """
    Find the first element with itemprop=prop.
    attr='text'    → returns .get_text(strip=True)
    attr='content' → returns the content= attribute (meta tags)
    attr='href'    → returns the href= attribute
    attr='src'     → returns the src= attribute
    """
    tag = soup.find(itemprop=prop)
    if tag is None:
        return ""
    if attr == "text":
        return tag.get_text(strip=True)
    return tag.get(attr, "").strip()


def _itemprop_all(soup, prop: str) -> list:
    """Return get_text for ALL elements with itemprop=prop."""
    return [t.get_text(strip=True) for t in soup.find_all(itemprop=prop)]


# ─────────────────────────────────────────────────────────────────────────────
#  Logo detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_logo(soup, source: str = "") -> str:
    """
    Multi-strategy logo detection.
    Returns the img tag HTML string, or "" if not found.
    """
    source_lower = source.lower()

    # ── Strategy 0 (NEW): schema.org itemprop="image" ─────────────────────
    # brownbook and askmap use microdata; the primary image IS the logo.
    if any(d in source_lower for d in ("brownbook", "askmap")):
        img = soup.find("img", itemprop="image")
        if img:
            srcs = _all_srcs(img)
            if srcs and not _is_tiny(img, 20):
                return str(img)
        # Also check meta itemprop="image" (content= attribute)
        meta_img = soup.find("meta", itemprop="image")
        if meta_img:
            content = meta_img.get("content", "").strip()
            if content and content.startswith("http"):
                return f'<img src="{content}" data-source="itemprop:image">'

    # ── Strategy 0b (NEW): JSON-LD logo field ─────────────────────────────
    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(ld_blocks, "LocalBusiness", "Organization", "Store", "Restaurant")
    if ld:
        logo_val = ld.get("logo", "")
        if isinstance(logo_val, dict):
            logo_val = logo_val.get("url", logo_val.get("contentUrl", ""))
        if isinstance(logo_val, str) and logo_val.startswith("http"):
            return f'<img src="{logo_val}" data-source="json-ld:logo">'

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

    # ── Strategy 4: overlay/circular logo (enrollbusiness pattern) ────────
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

    # ── Strategy 5: enrollbusiness hero second-img is the overlay logo ────
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
                        return str(img)

    # ── Strategy 6: meta og:image as last resort ──────────────────────────
    og = (soup.find("meta", property="og:image") or
          soup.find("meta", attrs={"name": "og:image"}))
    if og and og.get("content", "").startswith("http"):
        return f'<img src="{og["content"]}" data-source="og:image">'

    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Photo detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_photos(soup, source: str = "") -> bool:
    """
    Multi-strategy photo detection. Returns True if business photos exist.
    """
    source_lower = source.lower()

    # ── Strategy 0 (NEW): JSON-LD image array ────────────────────────────
    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(ld_blocks, "LocalBusiness", "Organization", "Store", "Restaurant")
    if ld:
        images = ld.get("image", [])
        if isinstance(images, str) and images.startswith("http"):
            return True
        if isinstance(images, list) and len(images) > 0:
            return True

    # ── Strategy 1: gallery / photo section containers ────────────────────
    for container in soup.find_all(["div", "section", "ul", "figure"]):
        cls = _cls_str(container)
        if _PHOTO_CLASS_PATTERNS.search(cls):
            imgs = container.find_all("img")
            for img in imgs:
                srcs = _all_srcs(img)
                if srcs and not _is_tiny(img, 40) and not _is_ui_image(srcs[0]):
                    return True
            style = container.get("style", "")
            if "background" in style.lower() and "url(" in style.lower():
                return True

    # ── Strategy 2: CSS background-image on any element ───────────────────
    bg_urls = _css_background_images(soup)
    for url in bg_urls:
        if not _is_ui_image(url):
            return True

    # ── Strategy 3: hero / banner container ───────────────────────────────
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

    # ── Strategy 5: Photos tab / link ────────────────────────────────────
    for a in soup.find_all("a"):
        txt = a.get_text(strip=True).lower()
        href = a.get("href", "").lower()
        if txt in ("photos", "photo", "gallery", "images") or "photo" in href:
            if not any(d in href for d in ("google", "facebook", "twitter", "instagram")):
                return True

    # ── Strategy 6: og:image ──────────────────────────────────────────────
    og = (soup.find("meta", property="og:image") or
          soup.find("meta", attrs={"name": "og:image"}))
    if og and og.get("content", "").startswith("http"):
        og_url = og["content"]
        if not _is_ui_image(og_url):
            return True

    # ── Strategy 7: any two sufficiently large, non-UI <img> tags ─────────
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
        if any(x in src.lower() for x in ("favicon", "sprite", "icon-", "-icon")):
            continue
        large_img_count += 1
        if large_img_count >= 2:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helper: "good" description predicate
# ─────────────────────────────────────────────────────────────────────────────

_BAD_DESC_PHRASES = (
    "payment methods", "last update", "general information",
    "write a review", "sign up", "site map", "privacy policy",
    "terms of", "cookie", "copyright", "all rights reserved",
    "related companies", "opening hours", "phone number",
    "categories", "social", "directions", "get directions",
    "claim this", "report an error", "edit this",
    "add to favorites", "share this", "print this",
    "follow us", "contact us", "send message",
)


def _good_desc(text: str, min_len: int = 40) -> bool:
    if len(text) < min_len:
        return False
    tl = text.lower()
    return not any(b in tl for b in _BAD_DESC_PHRASES)


def _longest_good_para(soup) -> str:
    """Generic fallback: longest visible <p> that passes _good_desc."""
    best = ""
    for p in soup.find_all("p"):
        if _is_hidden_tag(p):
            continue
        text = p.get_text(separator=" ", strip=True)
        if _good_desc(text) and len(text) > len(best):
            best = text
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Per-domain extractors — description, website, hours, social
# ─────────────────────────────────────────────────────────────────────────────

# Common "skip" domains for website URL extraction
_SKIP_DOMAINS = (
    "enrollbusiness.com", "nearfinderus.com", "hotfrog.com",
    "brownbook.net", "freelistingusa.com", "smallbusinessusa.com",
    "askmap.net", "google.com", "facebook.com", "instagram.com",
    "twitter.com", "linkedin.com", "whatsapp.com", "youtube.com",
    "yelp.com",
)

_SOCIAL_DOMAINS = (
    "facebook.com", "instagram.com", "linkedin.com",
    "twitter.com", "x.com", "youtube.com", "tiktok.com",
    "whatsapp.com", "pinterest.com",
)


def _first_external_link(soup) -> str:
    """Return the first <a href> that is not in _SKIP_DOMAINS."""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
            return href
    return ""


def _social_links_generic(soup) -> str:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(s in href for s in _SOCIAL_DOMAINS):
            links.append(href)
    return ", ".join(dict.fromkeys(links))


# ── askmap.net ───────────────────────────────────────────────────────────────

def _extract_askmap(soup) -> dict:
    """
    askmap uses schema.org microdata throughout.
    Almost everything is in itemprop attributes.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": ""}

    # Description
    desc = _itemprop(soup, "description")
    if _good_desc(desc):
        out["description_text"] = desc
    else:
        out["description_text"] = _longest_good_para(soup)

    # Website: itemprop="url" first, then rel="nofollow" external links,
    # then generic fallback. askmap wraps the external website in a link
    # with rel="nofollow external" or class containing "website".
    url_tag = soup.find(itemprop="url")
    if url_tag:
        href = url_tag.get("href", "") or url_tag.get("content", "")
        if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
            out["website_url"] = href
    if not out["website_url"]:
        for a in soup.find_all("a", href=True, rel=True):
            rel = " ".join(a.get("rel", []))
            if "nofollow" in rel or "external" in rel:
                href = a["href"]
                if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
                    out["website_url"] = href
                    break
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

    # Hours: itemprop="openingHours" or itemprop="openingHoursSpecification"
    hours_tags = soup.find_all(itemprop="openingHours")
    if hours_tags:
        out["hours"] = "; ".join(t.get_text(strip=True) for t in hours_tags
                                  if t.get_text(strip=True))
    if not out["hours"]:
        spec_tags = soup.find_all(itemprop="openingHoursSpecification")
        if spec_tags:
            out["hours"] = "; ".join(t.get_text(separator=" ", strip=True)
                                      for t in spec_tags if t.get_text(strip=True))

    # Social + GBP
    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── brownbook.net ─────────────────────────────────────────────────────────────

def _extract_brownbook(soup) -> dict:
    """
    brownbook uses schema.org microdata AND sometimes JSON-LD.
    Itemprop attributes are the most reliable extraction path.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": ""}

    # ── Description ──
    # Priority 1: itemprop="description"
    desc = _itemprop(soup, "description")
    if _good_desc(desc):
        out["description_text"] = desc
    else:
        # Priority 2: div/p with class containing "description" or "about"
        for tag in soup.find_all(["div", "p", "section"]):
            cls = _cls_str(tag)
            if re.search(r"(description|about|overview|summary)", cls):
                text = tag.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # ── Website URL ──
    # Priority 1: itemprop="url"
    url_tag = soup.find(itemprop="url")
    if url_tag:
        href = url_tag.get("href", "") or url_tag.get("content", "")
        if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
            out["website_url"] = href

    # Priority 2: <a class="website"> or text "Visit Website"
    if not out["website_url"]:
        website_label = re.compile(r"(visit website|website|web site|official site)", re.I)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            txt = a.get_text(strip=True)
            cls = _cls_str(a)
            if (website_label.search(txt) or "website" in cls) and href.startswith("http"):
                if not any(s in href for s in _SKIP_DOMAINS):
                    out["website_url"] = href
                    break

    # Priority 3: any outbound link
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

    # ── Hours ──
    # openingHours microdata or a div/section labeled "Opening Hours"
    hours_tags = soup.find_all(itemprop="openingHours")
    if hours_tags:
        out["hours"] = "; ".join(t.get("content", t.get_text(strip=True))
                                  for t in hours_tags if t.get_text(strip=True))
    if not out["hours"]:
        for tag in soup.find_all(["div", "section", "table"]):
            cls = _cls_str(tag)
            if re.search(r"(opening.?hours|working.?hours|business.?hours|hours)", cls):
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    out["hours"] = text
                    break

    # ── Social + GBP ──
    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── freelistingusa.com ───────────────────────────────────────────────────────

def _extract_freelistingusa(soup) -> dict:
    """
    freelistingusa uses a mix of custom CSS classes and schema.org microdata.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": ""}

    # ── Description ──
    # Priority 1: div.listing-description, div.description, p.description
    for tag in soup.find_all(["div", "p", "section"]):
        cls = _cls_str(tag)
        if re.search(r"(listing[-_]?description|business[-_]?description|"
                     r"description|about[-_]?us|about|overview)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if _good_desc(text):
                out["description_text"] = text
                break
    # Priority 2: itemprop="description"
    if not out["description_text"]:
        desc = _itemprop(soup, "description")
        if _good_desc(desc):
            out["description_text"] = desc
    # Priority 3: heading "About" or "Description" → next sibling
    if not out["description_text"]:
        for h in soup.find_all(["h2", "h3", "h4", "strong"]):
            htxt = h.get_text(strip=True).lower()
            if htxt in ("about", "description", "about us", "overview", "about the business"):
                for sib in h.find_next_siblings(["p", "div"]):
                    text = sib.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
                if out["description_text"]:
                    break
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # ── Website URL ──
    # Priority 1: div.website a, or a[class*=website], or itemprop="url"
    for tag in soup.find_all(["div", "p", "li", "span"]):
        cls = _cls_str(tag)
        if re.search(r"(website|web[-_]?site|official[-_]?site)", cls):
            a = tag.find("a", href=True)
            if a:
                href = a["href"]
                if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
                    out["website_url"] = href
                    break
    if not out["website_url"]:
        url_tag = soup.find(itemprop="url")
        if url_tag:
            href = url_tag.get("href", "") or url_tag.get("content", "")
            if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
                out["website_url"] = href
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

    # ── Hours ──
    for tag in soup.find_all(["div", "section", "table", "ul"]):
        cls = _cls_str(tag)
        if re.search(r"(hours|working[-_]?hours|open[-_]?hours|schedule)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                out["hours"] = text
                break
    if not out["hours"]:
        hours_tags = _itemprop_all(soup, "openingHours")
        if hours_tags:
            out["hours"] = "; ".join(h for h in hours_tags if h)

    # ── Social + GBP ──
    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── hotfrog.com ───────────────────────────────────────────────────────────────

def _extract_hotfrog(soup) -> dict:
    """
    Hotfrog is a React SPA. All key data is in JSON-LD
    (<script type="application/ld+json">). DOM is a fallback.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": ""}

    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(ld_blocks, "LocalBusiness", "Organization", "Store",
                  "Restaurant", "MedicalBusiness", "HealthAndBeautyBusiness",
                  "LegalService", "HomeAndConstructionBusiness")

    if ld:
        # Description
        desc = ld.get("description", "")
        if _good_desc(desc):
            out["description_text"] = desc

        # Website
        url = ld.get("url", "")
        if url and not any(s in url for s in _SKIP_DOMAINS):
            out["website_url"] = url

        # Hours: openingHours is a list of strings like "Mo-Fr 09:00-17:00"
        hours_val = ld.get("openingHours", [])
        if isinstance(hours_val, list):
            out["hours"] = "; ".join(hours_val)
        elif isinstance(hours_val, str):
            out["hours"] = hours_val

        # openingHoursSpecification (more detailed)
        if not out["hours"]:
            specs = ld.get("openingHoursSpecification", [])
            if isinstance(specs, list):
                parts = []
                for spec in specs:
                    days = spec.get("dayOfWeek", "")
                    if isinstance(days, list):
                        days = ", ".join(days)
                    opens  = spec.get("opens", "")
                    closes = spec.get("closes", "")
                    if days:
                        parts.append(f"{days}: {opens}–{closes}".strip())
                out["hours"] = "; ".join(parts)

    # Fallback description from DOM if JSON-LD was empty / bad
    if not out["description_text"]:
        for tag in soup.find_all(["div", "p", "section"]):
            cls = _cls_str(tag)
            if re.search(r"(description|about|overview|summary|bio)", cls):
                text = tag.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # Fallback website from DOM
    if not out["website_url"]:
        label_re = re.compile(r"(visit website|website|web site|official site)", re.I)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (label_re.search(a.get_text(strip=True))
                    and href.startswith("http")
                    and not any(s in href for s in _SKIP_DOMAINS)):
                out["website_url"] = href
                break
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

    # Social (hotfrog sometimes has social icons in DOM even when JSON-LD is used)
    soc = _social_links_generic(soup)
    # Also check JSON-LD sameAs
    same_as = ld.get("sameAs", []) if ld else []
    if isinstance(same_as, str):
        same_as = [same_as]
    for url in same_as:
        if any(s in url for s in _SOCIAL_DOMAINS) and url not in soc:
            soc = (soc + ", " + url).strip(", ")
    out["social_links"] = soc

    # GBP
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── smallbusinessusa.com ──────────────────────────────────────────────────────

def _extract_smallbusinessusa(soup) -> dict:
    """
    smallbusinessusa uses recognisable CSS class names; no schema.org.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": ""}

    # ── Description ──
    # Priority 1: div.business-description, div.about, div.company-description
    for tag in soup.find_all(["div", "p", "section", "article"]):
        cls = _cls_str(tag)
        if re.search(r"(business[-_]?description|company[-_]?description|"
                     r"about[-_]?us|about|overview|summary|bio[-_]?text|"
                     r"listing[-_]?desc)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if _good_desc(text):
                out["description_text"] = text
                break
    # Priority 2: heading "About" → next sibling
    if not out["description_text"]:
        for h in soup.find_all(["h2", "h3", "h4", "strong"]):
            htxt = h.get_text(strip=True).lower()
            if htxt in ("about", "about us", "description", "overview"):
                for sib in h.find_next_siblings(["p", "div"]):
                    text = sib.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
                if out["description_text"]:
                    break
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # ── Website URL ──
    # Priority 1: div/li with class containing "website"
    for tag in soup.find_all(["div", "li", "p", "span"]):
        cls = _cls_str(tag)
        if "website" in cls:
            a = tag.find("a", href=True)
            if a:
                href = a["href"]
                if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
                    out["website_url"] = href
                    break
    # Priority 2: button/a labeled "Visit Website"
    if not out["website_url"]:
        label_re = re.compile(r"(visit website|website|official site|web site)", re.I)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (label_re.search(a.get_text(strip=True))
                    and href.startswith("http")
                    and not any(s in href for s in _SKIP_DOMAINS)):
                out["website_url"] = href
                break
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

    # ── Hours ──
    for tag in soup.find_all(["div", "section", "table", "ul"]):
        cls = _cls_str(tag)
        if re.search(r"(hours|working[-_]?hours|business[-_]?hours|schedule|open)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) < 500:
                out["hours"] = text
                break

    # ── Social + GBP ──
    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── nearfinderus.com  (already existed — kept + improved) ────────────────────

def _extract_nearfinderus(soup) -> dict:
    from urllib.parse import unquote
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": ""}

    # Description: P1 div.mt-4 > visible p
    for div in soup.find_all("div"):
        if "mt-4" in " ".join(div.get("class", [])):
            for p in div.find_all("p"):
                if not _is_hidden_tag(p):
                    text = p.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
        if out["description_text"]:
            break
    # P2: heading "More about" → next sibling
    if not out["description_text"]:
        for h in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            if "more about" in h.get_text().lower():
                for sib in h.find_next_siblings(["p", "div"]):
                    text = sib.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
                if out["description_text"]:
                    break
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # Website + social via redirect wrapper
    social_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/empresa/redirect?url=" in href:
            raw = href.split("url=")[1].split("&")[0]
            decoded = unquote(raw)
            if any(s in decoded for s in _SOCIAL_DOMAINS):
                social_links.append(decoded)
            elif not any(s in decoded for s in ("whatsapp.com", "nearfinderus.com")):
                if not out["website_url"]:
                    out["website_url"] = decoded
        elif href.startswith("http") and not out["website_url"]:
            if not any(s in href for s in _SKIP_DOMAINS):
                out["website_url"] = href
    out["social_links"] = ", ".join(dict.fromkeys(social_links))

    # Hours
    for tag in soup.find_all(["div", "section", "table"]):
        cls = _cls_str(tag)
        if re.search(r"(hours|schedule|working|opening)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) < 400:
                out["hours"] = text
                break

    # GBP
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── us.enrollbusiness.com (already existed — kept + improved) ────────────────

def _extract_enrollbusiness(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": ""}

    # Description P1: section/div labeled About/Description
    for tag in soup.find_all(["div", "section", "article"]):
        cls = _cls_str(tag)
        if re.search(r"(about|description|overview|summary|info[-_]?text)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if _good_desc(text):
                out["description_text"] = text
                break
    # P2: heading → next sibling
    if not out["description_text"]:
        for h in soup.find_all(["h1","h2","h3","h4","h5","strong"]):
            htxt = h.get_text(strip=True).lower()
            if htxt in ("about", "description", "about us", "overview"):
                for sib in h.find_next_siblings(["p", "div"]):
                    text = sib.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
                if out["description_text"]:
                    break
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # Website: "Visit Website" label or any outbound link
    website_labels = re.compile(r"(visit website|website|web site|official site|homepage)", re.I)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        txt = a.get_text(strip=True)
        if website_labels.search(txt) and href.startswith("http"):
            if not any(s in href for s in _SKIP_DOMAINS):
                out["website_url"] = href
                break
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

    # Hours: look for a section/div labeled Hours
    for tag in soup.find_all(["div", "section", "table"]):
        cls = _cls_str(tag)
        if re.search(r"(hours|working[-_]?hours|business[-_]?hours|schedule)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) < 400:
                out["hours"] = text
                break

    # Social + GBP
    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Domain router
# ─────────────────────────────────────────────────────────────────────────────

def _route_domain(source_lower: str):
    """Return the correct per-domain extractor function, or None for generic."""
    if "askmap" in source_lower:
        return _extract_askmap
    if "brownbook" in source_lower:
        return _extract_brownbook
    if "freelistingusa" in source_lower:
        return _extract_freelistingusa
    if "hotfrog" in source_lower:
        return _extract_hotfrog
    if "smallbusinessusa" in source_lower:
        return _extract_smallbusinessusa
    if "nearfinderus" in source_lower:
        return _extract_nearfinderus
    if "enrollbusiness" in source_lower:
        return _extract_enrollbusiness
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Main pre-extraction function
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_hints(page_html: str, page_text: str, source: str = "") -> dict:
    """
    Structurally extract all fields from page HTML using BeautifulSoup.
    Returns a hints dict with confirmed values passed to the AI as authoritative.

    New in v3:
      - hours field added
      - per-domain extractor routing for all 7 domains
      - JSON-LD extraction layer (hotfrog, brownbook, freelistingusa)
      - schema.org microdata extraction (askmap, brownbook)
    """
    hints = {
        "logo_html":        "",
        "logo_confirmed":   False,
        "photos_confirmed": False,
        "description_text": "",
        "website_url":      "",
        "social_links":     "",
        "gbp_link":         "",
        "hours":            "",         # NEW in v3
    }
    if not page_html:
        return hints

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page_html, "html.parser")
        source_lower = source.lower()

        # ── LOGO ─────────────────────────────────────────────────────────
        logo_html = _detect_logo(soup, source)
        if logo_html:
            hints["logo_html"]      = logo_html
            hints["logo_confirmed"] = True

        # ── PHOTOS ───────────────────────────────────────────────────────
        hints["photos_confirmed"] = _detect_photos(soup, source)

        # ── TEXT FIELDS via per-domain router ─────────────────────────────
        extractor = _route_domain(source_lower)
        if extractor:
            domain_out = extractor(soup)
        else:
            # Generic fallback for unknown domains
            domain_out = {
                "description_text": _longest_good_para(soup),
                "website_url":      _first_external_link(soup),
                "hours":            "",
                "social_links":     _social_links_generic(soup),
                "gbp_link":         "",
            }

        hints["description_text"] = domain_out.get("description_text", "")
        hints["website_url"]      = domain_out.get("website_url", "")
        hints["hours"]            = domain_out.get("hours", "")
        hints["social_links"]     = domain_out.get("social_links", "")
        hints["gbp_link"]         = domain_out.get("gbp_link", "")

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
    hints = _extract_page_hints(page_html, page_text, source)

    all_img_tags = re.findall(r'<img[^>]*>', page_html, flags=re.IGNORECASE)
    img_section  = "\n".join(all_img_tags[:100]) if all_img_tags else ""

    bg_urls = re.findall(
        r"background(?:-image)?\s*:\s*url\(['\"]?([^'\"\)]+)['\"]?\)",
        page_html, re.IGNORECASE,
    )
    bg_section = "\n".join(bg_urls[:20]) if bg_urls else ""

    if len(page_html) > 20000:
        html_snippet = page_html[:15000] + "\n\n…[middle omitted]…\n\n" + page_html[-5000:]
    else:
        html_snippet = page_html

    # ── Build pre-extracted facts section ────────────────────────────────
    pre_extracted_facts = []

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
            f'DESCRIPTION TEXT (confirmed from page HTML — use verbatim):\n'
            f'{hints["description_text"]}'
        )
    if hints["website_url"]:
        pre_extracted_facts.append(
            f'WEBSITE URL (confirmed from page HTML — use as-is):\n'
            f'{hints["website_url"]}'
        )
    if hints["hours"]:
        pre_extracted_facts.append(
            f'HOURS (confirmed from page HTML — use as-is):\n'
            f'{hints["hours"]}'
        )
    if hints["social_links"]:
        pre_extracted_facts.append(
            f'SOCIAL MEDIA LINKS (confirmed from page HTML):\n{hints["social_links"]}'
        )
    if hints["gbp_link"]:
        pre_extracted_facts.append(
            f'GBP LINK (confirmed from page HTML):\n{hints["gbp_link"]}'
        )

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
            f"CSS BACKGROUND-IMAGE URLs (real images — if business photos or logo, "
            f"treat as PRESENT):\n{bg_section}"
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
            field_rules.append('- "State": state / region as shown.')
        elif f == "Zipcode":
            field_rules.append('- "Zipcode": postal/zip code.')
        elif f == "Country":
            field_rules.append(
                '- "Country": country name or code. '
                'A two-letter ISO code is a valid answer.'
            )
        elif f == "Category":
            field_rules.append('- "Category": business type / industry category shown.')
        elif f == "Keywords":
            field_rules.append(
                '- "Keywords": tags, keywords, or labels for the business. '
                'Return comma-separated.'
            )
        elif f == "Description":
            field_rules.append(
                '- "Description": the business\'s own descriptive text.\n'
                '  If DESCRIPTION TEXT appears in PRE-EXTRACTED FIELDS above, use it verbatim.\n'
                '  Otherwise find prose paragraphs describing what the business does.\n'
                '  EXCLUDE: navigation text, review prompts, "Payment methods", '
                '"Last update", footer text.\n'
                '  Return null only if no genuine description exists.'
            )
        elif f == "Hours":
            field_rules.append(
                '- "Hours": operating hours.\n'
                '  If HOURS appears in PRE-EXTRACTED FIELDS above, use it directly.\n'
                '  Otherwise find opening-hours or working-hours sections.\n'
                '  Format: e.g. "Mon-Fri 9am-5pm" or "Monday - Sunday: 24 Hours Open".'
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
                '  ★ If "LOGO CONFIRMED PRESENT" appears in PRE-EXTRACTED FIELDS → '
                'return "PRESENT" immediately.\n'
                '  Otherwise check ALL <img> TAGS for logo/brand/avatar/profile src or class.\n'
                '  Also check CSS BACKGROUND-IMAGE URLs.\n'
                '  Return "PRESENT" if found, null if truly absent.'
            )
        elif f == "Photos":
            field_rules.append(
                '- "Photos": do business photos exist on this page?\n'
                '  ★ If "PHOTOS CONFIRMED PRESENT" appears in PRE-EXTRACTED FIELDS → '
                'return "PRESENT" immediately.\n'
                '  Otherwise check for hero/banner images, gallery sections, '
                'CSS backgrounds, Photos tabs, og:image.\n'
                '  Return null ONLY if absolutely no images of any kind exist.'
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

        # ── POST-PROCESS: authoritative overrides from pre-extraction ─────
        # These survive AI hallucination and context-window truncation.
        hints = _extract_page_hints(page_html, page_text, source)

        if hints["logo_confirmed"] and "Logo" in fields:
            extracted["Logo"] = "PRESENT"
        if hints["photos_confirmed"] and "Photos" in fields:
            extracted["Photos"] = "PRESENT"
        # Only override text fields if AI returned null/empty AND we have a value
        if hints["description_text"] and "Description" in fields:
            if not extracted.get("Description"):
                extracted["Description"] = hints["description_text"]
        if hints["website_url"] and "Website URL" in fields:
            if not extracted.get("Website URL"):
                extracted["Website URL"] = hints["website_url"]
        if hints["hours"] and "Hours" in fields:
            if not extracted.get("Hours"):
                extracted["Hours"] = hints["hours"]
        if hints["social_links"] and "Social Media Links" in fields:
            if not extracted.get("Social Media Links"):
                extracted["Social Media Links"] = hints["social_links"]
        if hints["gbp_link"] and "GBP Link" in fields:
            if not extracted.get("GBP Link"):
                extracted["GBP Link"] = hints["gbp_link"]

    except json.JSONDecodeError as e:
        extracted = {"_parse_error": str(e), "_raw": raw[:800], "_model": model_used}
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

"""
ai_extractor.py
Sends scraped page content to Google Gemini and extracts
business fields as JSON — including visual fields (Logo, Photos).
No hardcoded layout assumptions. Gemini searches the whole page.

v7 — Bug-fixes over v6:
  - nearfinderus — Keywords: now ONLY extracted from <meta name="keywords">.
      If the page has no meta keywords tag, keywords stays empty and Gemini is
      told explicitly NOT to infer/guess keywords (return null). Previously
      Gemini was hallucinating keywords from the page text.
  - nearfinderus — Description: added Priority 5 which scrapes the FULL text
      from the "More about BUSINESS in City" section rendered at the bottom of
      the page. This avoids the "See More" truncation that cut descriptions at
      ~200 chars. Also added a reversed-<p> scan as a secondary fallback.
  - nearfinderus — Hours: the day-pattern fallback now rejects any candidate
      that (a) contains the word "review" but no HH:MM time, or (b) contains
      day-names but no actual time info (no HH:MM, no am/pm, no open/closed).
      This prevents the "Reviews|0.0 (0 reviews)|Write a review" block from
      being returned as hours.
  - enrollbusiness — all-NULL fallback: added _parse_enrollbusiness_slug() that
      extracts Name, City, State, Zipcode from the URL path when JS rendering
      yields zero content. Applied as absolute last resort in extract_fields().
  - build_prompt — Keywords rule updated: if no KEYWORDS hint is present,
      Gemini is now explicitly told to return null (not to infer from page text).
  - scraper.py config note: enrollbusiness wait should be 25000ms with selector
      ".profile-home, h1, [class*='business-name']" for better JS hydration.
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

# address-like string detector — used to reject false descriptions
_ADDRESS_LIKE = re.compile(
    r"^\s*(address\s*:|phone\s*:|\d+\s+\w+.*\b(blvd|st|ave|rd|ln|dr|way|ct|pl)\b)",
    re.IGNORECASE,
)

_PLACEHOLDER_HOURS = re.compile(
    r"^0{1,2}:0{2}\s*to\s*0{1,2}:0{2}$"
)

_PHONE_IN_TEXT = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")

_ROAD_TYPES = re.compile(
    r"\b(blvd|boulevard|street|st\b|avenue|ave\b|road\b|rd\b|lane\b|ln\b|"
    r"drive\b|dr\b|way\b|court\b|ct\b|place\b|pl\b|circle\b|cir\b|"
    r"parkway\b|pkwy\b|highway\b|hwy\b|suite\b|ste\b|floor\b|fl\b)\b",
    re.IGNORECASE,
)


def _all_srcs(tag) -> list:
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
#  JSON-LD extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_ld(soup) -> list:
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
    type_lower = [t.lower() for t in types_]
    for block in ld_blocks:
        bt = block.get("@type", "")
        if isinstance(bt, list):
            bt = " ".join(bt)
        if any(t in bt.lower() for t in type_lower):
            return block
    return {}


def _ld_address(ld: dict) -> dict:
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
#  schema.org microdata extractor
# ─────────────────────────────────────────────────────────────────────────────

def _itemprop(soup, prop: str, attr: str = "text") -> str:
    tag = soup.find(itemprop=prop)
    if tag is None:
        return ""
    if attr == "text":
        return tag.get_text(strip=True)
    return tag.get(attr, "").strip()


def _itemprop_all(soup, prop: str) -> list:
    return [t.get_text(strip=True) for t in soup.find_all(itemprop=prop)]


# ─────────────────────────────────────────────────────────────────────────────
#  Logo detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_logo(soup, source: str = "") -> str:
    source_lower = source.lower()

    if any(d in source_lower for d in ("brownbook", "askmap")):
        img = soup.find("img", itemprop="image")
        if img:
            srcs = _all_srcs(img)
            if srcs and not _is_tiny(img, 20):
                return str(img)
        meta_img = soup.find("meta", itemprop="image")
        if meta_img:
            content = meta_img.get("content", "").strip()
            if content and content.startswith("http"):
                return f'<img src="{content}" data-source="itemprop:image">'

    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(ld_blocks, "LocalBusiness", "Organization", "Store", "Restaurant")
    if ld:
        logo_val = ld.get("logo", "")
        if isinstance(logo_val, dict):
            logo_val = logo_val.get("url", logo_val.get("contentUrl", ""))
        if isinstance(logo_val, str) and logo_val.startswith("http"):
            return f'<img src="{logo_val}" data-source="json-ld:logo">'

    for img in soup.find_all("img"):
        for src in _all_srcs(img):
            if _LOGO_SRC_PATTERNS.search(src) and not _is_tiny(img, 20):
                return str(img)

    for img in soup.find_all("img"):
        img_cls = _cls_str(img)
        alt = img.get("alt", "").lower()
        if (re.search(r"(logo|brand|emblem|crest)", img_cls) or
                re.search(r"(logo|brand)", alt)):
            srcs = _all_srcs(img)
            if srcs and not _is_tiny(img, 20):
                return str(img)

    for img in soup.find_all("img"):
        anc = _ancestor_cls(img, depth=5)
        if _LOGO_CLASS_PATTERNS.search(anc):
            srcs = _all_srcs(img)
            if srcs and not _is_tiny(img, 20) and not _is_ui_image(srcs[0]):
                return str(img)

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

    og = (soup.find("meta", property="og:image") or
          soup.find("meta", attrs={"name": "og:image"}))
    if og and og.get("content", "").startswith("http"):
        return f'<img src="{og["content"]}" data-source="og:image">'

    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Photo detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_photos(soup, source: str = "") -> bool:
    source_lower = source.lower()

    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(ld_blocks, "LocalBusiness", "Organization", "Store", "Restaurant")
    if ld:
        images = ld.get("image", [])
        if isinstance(images, str) and images.startswith("http"):
            return True
        if isinstance(images, list) and len(images) > 0:
            return True

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

    bg_urls = _css_background_images(soup)
    for url in bg_urls:
        if not _is_ui_image(url):
            return True

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

    for img in soup.find_all("img"):
        for src in _all_srcs(img):
            if _PHOTO_SRC_PATTERNS.search(src) and not _is_ui_image(src):
                if not _is_tiny(img, 40):
                    return True

    for a in soup.find_all("a"):
        txt = a.get_text(strip=True).lower()
        href = a.get("href", "").lower()
        if txt in ("photos", "photo", "gallery", "images") or "photo" in href:
            if not any(d in href for d in ("google", "facebook", "twitter", "instagram")):
                return True

    og = (soup.find("meta", property="og:image") or
          soup.find("meta", attrs={"name": "og:image"}))
    if og and og.get("content", "").startswith("http"):
        og_url = og["content"]
        if not _is_ui_image(og_url):
            return True

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


def _good_desc(text: str, min_len: int = 60) -> bool:
    if len(text) < min_len:
        return False

    tl = text.lower()

    if any(b in tl for b in _BAD_DESC_PHRASES):
        return False

    if _ADDRESS_LIKE.match(text):
        return False

    normalised = re.sub(r"[–—]", " ", text)
    words = normalised.split()
    word_count = len(words)

    if word_count < 12 and _ROAD_TYPES.search(normalised):
        return False

    if re.search(r"\b[A-Z]{2}\s+\d{5}\b", normalised):
        return False

    if re.search(r"\b\d{5}\b", normalised):
        return False

    addr_tokens = re.findall(
        r"\b(\d{3,}|blvd|street|avenue|suite|ste|fl\b|zip|phone|tel)\b",
        normalised.lower(),
    )
    if word_count > 0 and len(addr_tokens) / word_count > 0.35:
        return False

    return True


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
#  Keywords cleaner
# ─────────────────────────────────────────────────────────────────────────────

_ADDRESS_TOKEN_PATTERNS = re.compile(
    r"^\s*(\d{3,}.*|.*\b(blvd|boulevard|street|avenue|ave|road|lane|drive|"
    r"way|court|place|circle|parkway|highway|suite|ste)\b.*|"
    r"\d{5}(-\d{4})?|[a-z]{2}\s+\d{5})\s*$",
    re.IGNORECASE,
)

_KEYWORD_NOISE = re.compile(
    r"^(address details|roadmap|satellite map|phone number|business hours|"
    r"trip planner?|travel|maps?|location|venue|place|trip)\s*$",
    re.IGNORECASE,
)


def _clean_keywords(raw_keywords: str, business_name: str = "") -> str:
    if not raw_keywords:
        return ""
    tokens = [t.strip() for t in raw_keywords.split(",") if t.strip()]
    bn_lower = business_name.lower().strip() if business_name else ""
    cleaned = []
    for tok in tokens:
        tl = tok.lower()
        if _ADDRESS_TOKEN_PATTERNS.match(tok):
            continue
        if _KEYWORD_NOISE.match(tok):
            continue
        if re.search(r"\b\d{5}\b", tok):
            continue
        if re.match(r"^[a-z\s]+,?\s+[a-z]{2}$", tok, re.IGNORECASE):
            continue
        if re.match(r"^[a-z]{2}$", tok, re.IGNORECASE):
            continue
        if re.match(r"^\d{5}(-\d{4})?$", tok):
            continue
        if bn_lower and tl == bn_lower:
            continue
        words_in_tok = tl.split()
        if len(words_in_tok) <= 2:
            service_words = re.compile(
                r"(service|repair|restoration|cleaning|plumbing|roofing|"
                r"contractor|construction|damage|emergency|water|fire|mold|"
                r"storm|biohazard|insurance|medical|legal|dental|auto|"
                r"electric|hvac|painting|flooring|landscape|pest|security|"
                r"catering|moving|storage|shipping|printing|design|marketing|"
                r"consulting|accounting|law|clinic|salon|spa|gym|fitness|"
                r"restaurant|cafe|hotel|retail|wholesale|import|export|"
                r"technology|software|hardware|network|cloud|digital)",
                re.IGNORECASE,
            )
            if not service_words.search(tl) and _ROAD_TYPES.search(tl) is None:
                if re.match(r"^[A-Z][a-z]+$", tok) or re.match(r"^[a-z]+$", tok):
                    if not service_words.search(tl):
                        continue
        cleaned.append(tok)
    return ", ".join(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared category helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_category_generic(soup) -> str:
    cat = _itemprop(soup, "category")
    if cat and len(cat) < 80:
        return cat

    ld_blocks = _extract_json_ld(soup)
    for ld in ld_blocks:
        for key in ("additionalType", "knowsAbout", "serviceType", "category"):
            val = ld.get(key, "")
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val and len(val) < 80:
                val = val.split("/")[-1].replace("-", " ").replace("_", " ")
                return val

    for nav in soup.find_all(["nav", "ol", "ul", "div"],
                              class_=re.compile(r"breadcrumb", re.I)):
        items = nav.find_all(["li", "a", "span"])
        texts = [i.get_text(strip=True) for i in items if i.get_text(strip=True)]
        if len(texts) >= 2:
            candidate = texts[-2]
            if 3 < len(candidate) < 60:
                return candidate

    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        kw = meta_kw.get("content", "").split(",")[0].strip()
        if kw and len(kw) < 60:
            return kw

    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Common skip domains
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_DOMAINS = (
    "enrollbusiness.com", "nearfinderus.com", "nearfinder.com",
    "hotfrog.com", "brownbook.net", "freelistingusa.com",
    "smallbusinessusa.com", "askmap.net", "google.com",
    "facebook.com", "instagram.com", "twitter.com", "linkedin.com",
    "whatsapp.com", "youtube.com", "yelp.com", "cloudflare.com",
)

_SOCIAL_DOMAINS = (
    "facebook.com", "instagram.com", "linkedin.com",
    "twitter.com", "x.com", "youtube.com", "tiktok.com",
    "whatsapp.com", "pinterest.com",
)


def _first_external_link(soup) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
#  Per-domain extractors
# ─────────────────────────────────────────────────────────────────────────────

# ── askmap.net ───────────────────────────────────────────────────────────────

def _extract_askmap(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": "",
           "keywords": ""}

    desc = _itemprop(soup, "description")
    if _good_desc(desc):
        out["description_text"] = desc

    if not out["description_text"]:
        for tag in soup.find_all(["div", "section", "p"]):
            cls = _cls_str(tag)
            if re.search(r"\b(info|about|description|overview|summary)\b", cls):
                if _is_hidden_tag(tag):
                    continue
                text = tag.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break

    if not out["description_text"]:
        for h in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            htxt = h.get_text(strip=True).lower()
            if htxt in ("info", "about", "description", "about us", "overview"):
                for sib in h.find_next_siblings(["p", "div"]):
                    text = sib.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
                if out["description_text"]:
                    break

    if not out["description_text"]:
        info_h = soup.find(["h3", "h4", "strong"], string=re.compile(r"^\s*Info\s*$", re.I))
        if info_h:
            for sib in info_h.find_next_siblings():
                text = sib.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break

    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    out["category"] = _extract_category_generic(soup)

    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        raw_kw = meta_kw.get("content", "")
        out["keywords"] = _clean_keywords(raw_kw)

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
        www_img = soup.find("img", src=re.compile(r"website|www|globe|web", re.I))
        if www_img:
            parent_a = www_img.find_parent("a")
            if parent_a:
                href = parent_a.get("href", "")
                if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
                    out["website_url"] = href
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

    hours_tags = soup.find_all(itemprop="openingHours")
    if hours_tags:
        out["hours"] = "; ".join(t.get_text(strip=True) for t in hours_tags
                                  if t.get_text(strip=True))
    if not out["hours"]:
        spec_tags = soup.find_all(itemprop="openingHoursSpecification")
        if spec_tags:
            out["hours"] = "; ".join(t.get_text(separator=" ", strip=True)
                                      for t in spec_tags if t.get_text(strip=True))

    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── brownbook.net ─────────────────────────────────────────────────────────────

def _extract_brownbook(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": "",
           "keywords": ""}

    desc = _itemprop(soup, "description")
    if _good_desc(desc):
        out["description_text"] = desc
    else:
        for tag in soup.find_all(["div", "p", "section"]):
            cls = _cls_str(tag)
            if re.search(r"(description|about|overview|summary)", cls):
                text = tag.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    out["category"] = _extract_category_generic(soup)

    url_tag = soup.find(itemprop="url")
    if url_tag:
        href = url_tag.get("href", "") or url_tag.get("content", "")
        if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
            out["website_url"] = href
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
    if not out["website_url"]:
        out["website_url"] = _first_external_link(soup)

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

    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    # Keywords from "Business tags" section
    for tag in soup.find_all(["div", "section"]):
        cls = _cls_str(tag)
        heading = tag.find(["h2", "h3", "h4", "strong", "b"])
        if heading and re.search(r"business\s*tags?", heading.get_text(strip=True), re.IGNORECASE):
            links = tag.find_all("a")
            kw_tokens = [a.get_text(strip=True) for a in links if a.get_text(strip=True)]
            if kw_tokens:
                out["keywords"] = _clean_keywords(", ".join(kw_tokens))
                break
    # Fallback: meta keywords
    if not out["keywords"]:
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw:
            out["keywords"] = _clean_keywords(meta_kw.get("content", ""))

    return out


# ── freelistingusa.com ───────────────────────────────────────────────────────

def _extract_freelistingusa(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": "",
           "keywords": ""}

    for tag in soup.find_all(["div", "p", "section"]):
        cls = _cls_str(tag)
        if re.search(r"(listing[-_]?description|business[-_]?description|"
                     r"description|about[-_]?us|about|overview)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if _good_desc(text):
                out["description_text"] = text
                break
    if not out["description_text"]:
        desc = _itemprop(soup, "description")
        if _good_desc(desc):
            out["description_text"] = desc
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

    out["category"] = _extract_category_generic(soup)

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

    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    # Keywords from "Tags:" label or tags section
    for tag in soup.find_all(["div", "p", "span", "li"]):
        cls = _cls_str(tag)
        txt = tag.get_text(separator=" ", strip=True)
        if re.search(r"^tags?\s*:", txt, re.IGNORECASE) or re.search(r"\btags?\b", cls):
            # Strip the "Tags:" prefix and extract remaining text
            kw_raw = re.sub(r"^tags?\s*:\s*", "", txt, flags=re.IGNORECASE).strip()
            if kw_raw:
                out["keywords"] = _clean_keywords(kw_raw)
                break
    # Also check for anchor tags inside a tags-labeled container
    if not out["keywords"]:
        for tag in soup.find_all(["div", "section", "p"]):
            cls = _cls_str(tag)
            if re.search(r"\btags?\b", cls):
                links = tag.find_all("a")
                kw_tokens = [a.get_text(strip=True) for a in links if a.get_text(strip=True)]
                if kw_tokens:
                    out["keywords"] = _clean_keywords(", ".join(kw_tokens))
                    break
    # Fallback: meta keywords
    if not out["keywords"]:
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw:
            out["keywords"] = _clean_keywords(meta_kw.get("content", ""))

    return out


# ── hotfrog.com ───────────────────────────────────────────────────────────────

def _extract_hotfrog(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": "",
           "keywords": ""}

    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(ld_blocks, "LocalBusiness", "Organization", "Store",
                  "Restaurant", "MedicalBusiness", "HealthAndBeautyBusiness",
                  "LegalService", "HomeAndConstructionBusiness")

    if ld:
        desc = ld.get("description", "")
        if _good_desc(desc):
            out["description_text"] = desc

        url = ld.get("url", "")
        if url and not any(s in url for s in _SKIP_DOMAINS):
            out["website_url"] = url

        hours_val = ld.get("openingHours", [])
        if isinstance(hours_val, list):
            out["hours"] = "; ".join(hours_val)
        elif isinstance(hours_val, str):
            out["hours"] = hours_val

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

        for key in ("additionalType", "knowsAbout", "serviceType"):
            val = ld.get(key, "")
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val:
                out["category"] = val.split("/")[-1].replace("-", " ")
                break

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

    if not out["category"]:
        out["category"] = _extract_category_generic(soup)

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

    soc = _social_links_generic(soup)
    same_as = ld.get("sameAs", []) if ld else []
    if isinstance(same_as, str):
        same_as = [same_as]
    for url in same_as:
        if any(s in url for s in _SOCIAL_DOMAINS) and url not in soc:
            soc = (soc + ", " + url).strip(", ")
    out["social_links"] = soc

    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    # Keywords from "Focal's Keywords" / "[Name]'s Keywords" section
    # Hotfrog uses pipe-separated keywords: "ChatGPT Ads | ChatGPT Ads Agency"
    for heading in soup.find_all(["h2", "h3", "h4", "strong", "b", "p"]):
        htxt = heading.get_text(strip=True)
        if re.search(r"keywords?", htxt, re.IGNORECASE):
            # Try sibling text or next element
            for sib in heading.find_next_siblings(["p", "div", "span", "ul"]):
                text = sib.get_text(separator=" | ", strip=True)
                if text and len(text) < 300:
                    # Convert pipe-separated to comma-separated
                    kw_raw = re.sub(r"\s*\|\s*", ", ", text)
                    cleaned = _clean_keywords(kw_raw)
                    if cleaned:
                        out["keywords"] = cleaned
                        break
            if out["keywords"]:
                break
    # Fallback: check JSON-LD keywords field
    if not out["keywords"] and ld:
        kw_val = ld.get("keywords", "")
        if isinstance(kw_val, list):
            kw_val = ", ".join(kw_val)
        if isinstance(kw_val, str) and kw_val:
            out["keywords"] = _clean_keywords(kw_val)
    # Fallback: meta keywords
    if not out["keywords"]:
        meta_kw = soup.find("meta", attrs={"name": "keywords"})
        if meta_kw:
            out["keywords"] = _clean_keywords(meta_kw.get("content", ""))

    return out


# ── smallbusinessusa.com ──────────────────────────────────────────────────────

def _extract_smallbusinessusa(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": ""}

    for tag in soup.find_all(["div", "p", "section", "article"]):
        cls = _cls_str(tag)
        if re.search(r"(business[-_]?description|company[-_]?description|"
                     r"about[-_]?us|about|overview|summary|bio[-_]?text|"
                     r"listing[-_]?desc)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if _good_desc(text):
                out["description_text"] = text
                break
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

    out["category"] = _extract_category_generic(soup)

    for tag in soup.find_all(["div", "li", "p", "span"]):
        cls = _cls_str(tag)
        if "website" in cls:
            a = tag.find("a", href=True)
            if a:
                href = a["href"]
                if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
                    out["website_url"] = href
                    break
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

    for tag in soup.find_all(["div", "section", "table", "ul"]):
        cls = _cls_str(tag)
        if re.search(r"(hours|working[-_]?hours|business[-_]?hours|schedule|open)", cls):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) < 500:
                out["hours"] = text
                break

    out["social_links"] = _social_links_generic(soup)
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── nearfinderus.com ─────────────────────────────────────────────────────────

def _extract_nearfinderus(soup) -> dict:
    """
    FIX v7:
      Keywords: ONLY from <meta name="keywords"> — never guessed.
      Description: added Priority 5 — scrapes the full text from the
        "More about BUSINESS in City" section at the bottom of the page,
        which avoids the "See More" clamp on the top snippet.
      Hours: day-pattern fallback now rejects text that contains "review"
        but no HH:MM time, and also rejects text with no real time info.
    """
    from urllib.parse import unquote
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": "",
           "keywords": ""}

    # ── Description ──

    # Priority 1: "More about" / "About" heading sibling paragraphs
    for h in soup.find_all(["h2", "h3", "h4", "strong", "b", "p"]):
        htxt = h.get_text(strip=True).lower()
        if re.search(r"more about|about\s+\w", htxt):
            for sib in h.find_next_siblings(["p", "div", "section"]):
                if _is_hidden_tag(sib):
                    continue
                text = sib.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break
            if not out["description_text"] and h.parent:
                for child in h.parent.find_all(["p", "div"], recursive=False):
                    if child == h:
                        continue
                    if _is_hidden_tag(child):
                        continue
                    text = child.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
            if out["description_text"]:
                break

    # Priority 2: div.mt-4 > p
    if not out["description_text"]:
        for div in soup.find_all("div"):
            classes = " ".join(div.get("class", []))
            if "mt-4" in classes:
                for p in div.find_all("p"):
                    if _is_hidden_tag(p):
                        continue
                    text = p.get_text(separator=" ", strip=True)
                    if _good_desc(text):
                        out["description_text"] = text
                        break
            if out["description_text"]:
                break

    # Priority 3: class-based description div
    if not out["description_text"]:
        for tag in soup.find_all(["div", "section", "article"]):
            cls = _cls_str(tag)
            if re.search(r"\b(description|about|overview|info)\b", cls):
                if _is_hidden_tag(tag):
                    continue
                text = tag.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break

    # Priority 4: longest good paragraph
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # Priority 5 (FIX v7): full text from the bottom "More about BUSINESS in City" section.
    # nearfinderus truncates the description at the top with a "See More" button,
    # but renders the COMPLETE text in a section near the bottom of the page.
    # We prefer the longer version when the current extraction is short (<300 chars).
    if not out["description_text"] or len(out["description_text"]) < 300:
        # Strategy A: find a heading containing "More about" anywhere in the page
        for heading_tag in soup.find_all(["h2", "h3", "h4", "strong"]):
            if re.search(r"more\s+about", heading_tag.get_text(strip=True), re.IGNORECASE):
                container = heading_tag.parent
                if container:
                    paragraphs = container.find_all("p")
                    combined = " ".join(
                        p.get_text(separator=" ", strip=True)
                        for p in paragraphs
                        if not _is_hidden_tag(p)
                    )
                    if combined and len(combined) > 50 and len(combined) > len(out["description_text"]):
                        out["description_text"] = combined
                        break
                    # Also try siblings of the heading
                    for sib in heading_tag.find_next_siblings(["p", "div"]):
                        if _is_hidden_tag(sib):
                            continue
                        text = sib.get_text(separator=" ", strip=True)
                        if text and len(text) > 50 and len(text) > len(out["description_text"]):
                            out["description_text"] = text
                            break

    # Strategy B (FIX v7): scan ALL <p> tags from the bottom up for a long description
    if not out["description_text"] or len(out["description_text"]) < 300:
        all_paras = soup.find_all("p")
        for p in reversed(all_paras):
            if _is_hidden_tag(p):
                continue
            text = p.get_text(separator=" ", strip=True)
            if text and len(text) > 100 and len(text) > len(out["description_text"]):
                out["description_text"] = text
                break

    # ── Category ──
    out["category"] = _extract_category_generic(soup)
    if not out["category"]:
        breadcrumb_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "category_" in href or "/category/" in href:
                txt = a.get_text(strip=True)
                if txt:
                    breadcrumb_links.append(txt)
        if breadcrumb_links:
            out["category"] = breadcrumb_links[-1]

    # ── Keywords (FIX v7: ONLY from meta tag — never guessed) ──
    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        raw_kw = meta_kw.get("content", "").strip()
        if raw_kw:
            out["keywords"] = _clean_keywords(raw_kw)
    # If no meta keywords tag or empty content → out["keywords"] stays ""
    # The build_prompt Keywords rule will tell Gemini to return null.

    # ── Website + Social via redirect wrapper ──
    social_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        redirect_match = re.search(r"/redirect\?url=([^&\s]+)", href)
        if redirect_match:
            decoded = unquote(redirect_match.group(1))
            if any(s in decoded for s in _SOCIAL_DOMAINS):
                social_links.append(decoded)
            elif not any(s in decoded for s in _SKIP_DOMAINS):
                if not out["website_url"]:
                    out["website_url"] = decoded
        elif href.startswith("http") and not out["website_url"]:
            if not any(s in href for s in _SKIP_DOMAINS):
                out["website_url"] = href
    out["social_links"] = ", ".join(dict.fromkeys(social_links))

    # ── Hours ──

    # Priority 1: itemprop="openingHours" content attribute
    oh_tags = soup.find_all(attrs={"itemprop": "openingHours"})
    if oh_tags:
        hours_list = []
        for t in oh_tags:
            val = t.get("content", "").strip() or t.get_text(strip=True)
            if not val:
                continue
            stripped = val.replace(" ", "")
            if re.search(r"0{1,2}:0{2}[-–to]+0{1,2}:0{2}", stripped):
                continue
            hours_list.append(val)
        if hours_list:
            out["hours"] = "; ".join(hours_list)

    # Priority 2: opening-hours table/list rows
    if not out["hours"]:
        for container in soup.find_all(["div", "section", "table", "ul"]):
            cls = _cls_str(container)
            # Skip address/contact containers
            if re.search(r"\baddress\b|\bcontact\b|\bphone\b", cls):
                continue
            if re.search(r"(hours|schedule|working|opening|open)", cls):
                rows = container.find_all("tr")
                if rows:
                    hours_rows = []
                    for row in rows:
                        cells = [td.get_text(strip=True)
                                 for td in row.find_all(["td", "th"])]
                        if len(cells) >= 2:
                            time_part = cells[1].strip()
                            if _PLACEHOLDER_HOURS.match(time_part):
                                continue
                            hours_rows.append(": ".join(cells[:2]))
                    if hours_rows:
                        candidate = "; ".join(hours_rows)
                        if not _PHONE_IN_TEXT.search(candidate):
                            out["hours"] = candidate
                        break
                else:
                    text = container.get_text(separator="|", strip=True)
                    placeholder_count = len(re.findall(r"0{1,2}:0{2}\s*to\s*0{1,2}:0{2}", text))
                    if text and len(text) < 600 and placeholder_count == 0:
                        if not _PHONE_IN_TEXT.search(text):
                            out["hours"] = text
                        break

    # Priority 3: day-name pattern fallback (FIX v7: reject reviews/non-time blocks)
    if not out["hours"]:
        day_pattern = re.compile(
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"mon|tue|wed|thu|fri|sat|sun)",
            re.IGNORECASE,
        )
        for tag in soup.find_all(["p", "div", "li", "span"]):
            text = tag.get_text(separator=" ", strip=True)
            if (day_pattern.search(text) and len(text) < 300
                    and not _is_hidden_tag(tag)):
                placeholder_count = len(
                    re.findall(r"0{1,2}:0{2}\s*to\s*0{1,2}:0{2}", text)
                )
                day_count = len(day_pattern.findall(text))
                if placeholder_count > 0 and placeholder_count >= day_count:
                    continue
                # Reject if phone number leaked in
                if _PHONE_IN_TEXT.search(text):
                    continue
                # FIX v7: reject reviews block (contains "review" but no HH:MM time)
                if (re.search(r"\breview", text, re.IGNORECASE) and
                        not re.search(r"\d{1,2}:\d{2}", text)):
                    continue
                # FIX v7: require actual time info — must have HH:MM OR open/closed/am/pm
                has_time_info = (
                    re.search(r"\d{1,2}:\d{2}", text) or
                    re.search(r"\b(open|closed|am|pm)\b", text, re.IGNORECASE)
                )
                if not has_time_info:
                    continue
                out["hours"] = text
                break

    # ── GBP ──
    for a in soup.find_all("a", href=True):
        if "google.com" in a["href"] and any(
                p in a["href"] for p in ("maps/place", "maps?q", "goo.gl")):
            out["gbp_link"] = a["href"]
            break

    return out


# ── us.enrollbusiness.com ─────────────────────────────────────────────────────

def _parse_enrollbusiness_slug(url: str) -> dict:
    """
    FIX v7: Last-resort URL slug parser for enrollbusiness.
    Parses Name, City, State, Zipcode from the URL path when JS rendering
    yields zero content.

    Example URL path segment:
      WrightWay-Emergency-Services-Nokomis-FL-34275
    → Name: "WrightWay Emergency Services"
      City: "Nokomis"
      State: "FL"
      Zipcode: "34275"

    Strategy:
      1. Extract the last path segment after /BusinessProfile/<id>/
      2. Split on hyphens.
      3. Detect a 5-digit ZIP at the end → Zipcode.
      4. Detect a 2-letter all-caps token just before ZIP → State.
      5. The token before State is likely City (single word).
      6. Everything before City → Name (rejoin with spaces).
    """
    out = {"name": "", "city": "", "state": "", "zip": ""}
    try:
        # Extract slug: everything after the numeric ID segment
        match = re.search(r"/BusinessProfile/\d+/([^/?#]+)", url)
        if not match:
            return out
        slug = match.group(1)
        parts = slug.split("-")
        if len(parts) < 3:
            return out

        # Work backwards: ZIP, State, City, then Name
        zip_code = ""
        state = ""
        city_parts = []
        name_parts = []

        idx = len(parts) - 1

        # Detect ZIP (5 digits)
        if re.match(r"^\d{5}$", parts[idx]):
            zip_code = parts[idx]
            idx -= 1

        # Detect State (2 uppercase letters)
        if idx >= 0 and re.match(r"^[A-Z]{2}$", parts[idx]):
            state = parts[idx]
            idx -= 1

        # Everything remaining is Name + City.
        # Heuristic: the last remaining token before state is City (proper noun, title-case).
        # We'll take the last 1 token as City (most enrollbusiness slugs have single-word city).
        remaining = parts[:idx + 1]
        if remaining:
            city_parts = [remaining[-1]]
            name_parts = remaining[:-1]

        out["name"] = " ".join(name_parts) if name_parts else ""
        out["city"] = " ".join(city_parts) if city_parts else ""
        out["state"] = state
        out["zip"] = zip_code
    except Exception:
        pass
    return out


def _extract_enrollbusiness(soup) -> dict:
    """
    FIX v6: Added JSON-LD + microdata pre-pass.
    FIX v7: Slug parsing is handled in extract_fields() as last resort.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": "",
           "name": "", "phone": "", "street": "", "city": "",
           "state": "", "zip": "", "country": ""}

    # ── JSON-LD structured data pre-pass ──
    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(ld_blocks, "LocalBusiness", "Organization", "Store",
                  "Restaurant", "Service", "ProfessionalService",
                  "HomeAndConstructionBusiness")
    if ld:
        if ld.get("name"):
            out["name"] = ld["name"]
        for key in ("telephone", "phone"):
            if ld.get(key):
                out["phone"] = ld[key]
                break
        addr = _ld_address(ld)
        out["street"]  = addr.get("street", "")
        out["city"]    = addr.get("city", "")
        out["state"]   = addr.get("state", "")
        out["zip"]     = addr.get("zip", "")
        out["country"] = addr.get("country", "")
        desc = ld.get("description", "")
        if _good_desc(desc):
            out["description_text"] = desc
        url = ld.get("url", "")
        if url and not any(s in url for s in _SKIP_DOMAINS):
            out["website_url"] = url
        hours_val = ld.get("openingHours", [])
        if isinstance(hours_val, list) and hours_val:
            out["hours"] = "; ".join(hours_val)
        elif isinstance(hours_val, str) and hours_val:
            out["hours"] = hours_val
        if not out["hours"]:
            specs = ld.get("openingHoursSpecification", [])
            if isinstance(specs, list):
                parts = []
                for spec in specs:
                    days = spec.get("dayOfWeek", "")
                    if isinstance(days, list):
                        days = ", ".join(d.split("/")[-1] for d in days)
                    opens  = spec.get("opens", "")
                    closes = spec.get("closes", "")
                    if days:
                        parts.append(f"{days}: {opens}–{closes}".strip())
                if parts:
                    out["hours"] = "; ".join(parts)
        for key in ("additionalType", "knowsAbout", "serviceType", "@type"):
            val = ld.get(key, "")
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val:
                out["category"] = val.split("/")[-1].replace("-", " ").replace("_", " ")
                break

    # ── Microdata (itemprop) pre-pass ──
    if not out["name"]:
        out["name"] = _itemprop(soup, "name")
    if not out["phone"]:
        out["phone"] = _itemprop(soup, "telephone")
    if not out["street"]:
        out["street"] = _itemprop(soup, "streetAddress")
    if not out["city"]:
        out["city"] = _itemprop(soup, "addressLocality")
    if not out["state"]:
        out["state"] = _itemprop(soup, "addressRegion")
    if not out["zip"]:
        out["zip"] = _itemprop(soup, "postalCode")
    if not out["country"]:
        out["country"] = _itemprop(soup, "addressCountry")
    if not out["description_text"]:
        desc = _itemprop(soup, "description")
        if _good_desc(desc):
            out["description_text"] = desc
    if not out["hours"]:
        oh_tags = soup.find_all(attrs={"itemprop": "openingHours"})
        if oh_tags:
            hours_list = []
            for t in oh_tags:
                val = t.get("content", "").strip() or t.get_text(strip=True)
                if val:
                    hours_list.append(val)
            if hours_list:
                out["hours"] = "; ".join(hours_list)

    # ── DOM-based extraction (fallback) ──
    if not out["description_text"]:
        for tag in soup.find_all(["div", "section", "article"]):
            cls = _cls_str(tag)
            if re.search(r"(about|description|overview|summary|info[-_]?text)", cls):
                text = tag.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break
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

    if not out["category"]:
        out["category"] = _extract_category_generic(soup)

    if not out["website_url"]:
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

    if not out["hours"]:
        for tag in soup.find_all(["div", "section", "table", "li"]):
            text = tag.get_text(separator=" ", strip=True)
            if re.search(r"(working hours|business hours|hours of operation|opening hours)", text, re.I):
                if len(text) < 500:
                    hours_text = re.sub(
                        r"^.*(working hours|business hours|hours of operation|opening hours)\s*[:\-]?\s*",
                        "", text, flags=re.IGNORECASE,
                    ).strip()
                    if hours_text:
                        out["hours"] = hours_text
                        break
    if not out["hours"]:
        for tag in soup.find_all(["div", "section", "table"]):
            cls = _cls_str(tag)
            if re.search(r"(hours|working[-_]?hours|business[-_]?hours|schedule)", cls):
                text = tag.get_text(separator=" ", strip=True)
                if text and len(text) < 400:
                    out["hours"] = text
                    break

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
#  Cloudflare challenge detector
# ─────────────────────────────────────────────────────────────────────────────

_CF_CHALLENGE_SIGNALS = (
    "cloudflare.com?utm_source=challenge",
    "cf_chl_",
    "cdn-cgi/challenge-platform",
    "Just a moment",
    "checking your browser",
    "DDoS protection by Cloudflare",
)


def _is_cloudflare_challenge(html: str, text: str) -> bool:
    combined = (html[:5000] + text[:2000]).lower()
    return any(s.lower() in combined for s in _CF_CHALLENGE_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pre-extraction function
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_hints(page_html: str, page_text: str, source: str = "") -> dict:
    hints = {
        "logo_html":        "",
        "logo_confirmed":   False,
        "photos_confirmed": False,
        "description_text": "",
        "website_url":      "",
        "social_links":     "",
        "gbp_link":         "",
        "hours":            "",
        "category":         "",
        "keywords":         "",
        "name":    "",
        "phone":   "",
        "street":  "",
        "city":    "",
        "state":   "",
        "zip":     "",
        "country": "",
        "cloudflare_blocked": False,
    }
    if not page_html:
        return hints

    if _is_cloudflare_challenge(page_html, page_text):
        hints["cloudflare_blocked"] = True
        return hints

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page_html, "html.parser")
        source_lower = source.lower()

        hints["logo_confirmed"]   = bool(_detect_logo(soup, source))
        hints["logo_html"]        = _detect_logo(soup, source) if hints["logo_confirmed"] else ""
        hints["photos_confirmed"] = _detect_photos(soup, source)

        extractor = _route_domain(source_lower)
        if extractor:
            domain_out = extractor(soup)
        else:
            domain_out = {
                "description_text": _longest_good_para(soup),
                "website_url":      _first_external_link(soup),
                "hours":            "",
                "social_links":     _social_links_generic(soup),
                "gbp_link":         "",
                "category":         _extract_category_generic(soup),
            }

        hints["description_text"] = domain_out.get("description_text", "")
        hints["website_url"]      = domain_out.get("website_url", "")
        hints["hours"]            = domain_out.get("hours", "")
        hints["social_links"]     = domain_out.get("social_links", "")
        hints["gbp_link"]         = domain_out.get("gbp_link", "")
        hints["category"]         = domain_out.get("category", "")
        hints["keywords"]         = domain_out.get("keywords", "")

        for f in ("name", "phone", "street", "city", "state", "zip", "country"):
            hints[f] = domain_out.get(f, "")

    except Exception:
        pass

    return hints


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(page_text: str, page_html: str, fields: list, source: str = "") -> str:
    hints = _extract_page_hints(page_html, page_text, source)

    if hints.get("cloudflare_blocked"):
        return (
            f'You are a business data extraction assistant.\n'
            f'This page returned a Cloudflare security challenge and contains NO real business data.\n'
            f'Return ONLY a JSON object with null for every field: {fields}\n'
            f'Example: {{{", ".join(repr(f)+": null" for f in fields)}}}'
        )

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
    if hints["category"]:
        pre_extracted_facts.append(
            f'CATEGORY (confirmed from page HTML — use as-is):\n{hints["category"]}'
        )
    if hints["keywords"]:
        pre_extracted_facts.append(
            f'KEYWORDS (cleaned from page meta-keywords — use as-is, NO address tokens):\n'
            f'{hints["keywords"]}'
        )
    for field_key, hint_key in [
        ("Name", "name"), ("Phone", "phone"), ("Street", "street"),
        ("City", "city"), ("State", "state"), ("Zipcode", "zip"),
        ("Country", "country"),
    ]:
        if hints.get(hint_key) and field_key in fields:
            pre_extracted_facts.append(
                f'{field_key.upper()} (from structured data — use as-is):\n{hints[hint_key]}'
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
                '  Otherwise find a "Visit Website" or external link — NOT the directory domain.\n'
                '  NEVER return a cloudflare.com URL.'
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
            field_rules.append(
                '- "Category": business type / industry category shown.\n'
                '  If CATEGORY appears in PRE-EXTRACTED FIELDS above, use it directly.'
            )
        elif f == "Keywords":
            field_rules.append(
                '- "Keywords": business-type tags and service keywords ONLY.\n'
                '  If KEYWORDS appears in PRE-EXTRACTED FIELDS above, use it directly.\n'
                '  If NO KEYWORDS hint is present in PRE-EXTRACTED FIELDS, you MUST\n'
                '  return null — this means the page has no meta keywords tag.\n'
                '  ABSOLUTELY DO NOT infer, guess, or extract keywords from page text,\n'
                '  description, category, title, or any other source.\n'
                '  The ONLY valid source for keywords is the PRE-EXTRACTED FIELDS hint.\n'
                '  If unsure, return null.\n'
                '  NEVER include address components (street names, city names, zip codes,\n'
                '  state abbreviations, road types like blvd/ave/st) in keywords.\n'
                '  NEVER include directory navigation terms (maps, location, trip, venue).'
            )
        elif f == "Description":
            field_rules.append(
                '- "Description": the business\'s own descriptive text.\n'
                '  If DESCRIPTION TEXT appears in PRE-EXTRACTED FIELDS above, use it verbatim.\n'
                '  Otherwise find prose paragraphs describing what the business does.\n'
                '  EXCLUDE: navigation text, review prompts, "Payment methods", '
                '"Last update", footer text, bare address strings, phone numbers.\n'
                '  NEVER return a street address as the description.\n'
                '  Return null only if no genuine description exists.'
            )
        elif f == "Hours":
            field_rules.append(
                '- "Hours": operating hours.\n'
                '  If HOURS appears in PRE-EXTRACTED FIELDS above, use it directly.\n'
                '  Otherwise find opening-hours or working-hours sections.\n'
                '  Format: e.g. "Mon-Fri 9am-5pm" or "Monday - Sunday: 24 Hours Open".\n'
                '  NEVER return hours where every day shows "00:00 to 00:00" — '
                'those are un-rendered placeholders; return null instead.\n'
                '  NEVER return an address or phone number as hours.\n'
                '  NEVER return review text (e.g. "0 reviews", "Write a review") as hours.'
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
- For Website URL: NEVER return a cloudflare.com URL — return null if only cloudflare URLs exist.
- For Description: NEVER return a bare address string or phone number as the description.
- For Hours: NEVER return a value where every day shows "00:00 to 00:00" — return null instead.
- For Hours: NEVER return an address, phone number, or review text as hours.
- For Keywords: if no KEYWORDS hint exists in PRE-EXTRACTED FIELDS, return null — do NOT infer.

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


def _all_null(extracted: dict, fields: list) -> bool:
    for f in fields:
        if extracted.get(f) not in (None, "", "null"):
            return False
    return True


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

        # ── POST-PROCESS: authoritative overrides ─────────────────────────
        hints = _extract_page_hints(page_html, page_text, source)

        if hints.get("cloudflare_blocked"):
            for f in fields:
                extracted[f] = None
            extracted["_cloudflare_blocked"] = True
            extracted["_model"] = model_used
            return extracted

        if hints["logo_confirmed"] and "Logo" in fields:
            extracted["Logo"] = "PRESENT"
        if hints["photos_confirmed"] and "Photos" in fields:
            extracted["Photos"] = "PRESENT"

        def _ai_empty(val) -> bool:
            return val in (None, "", "null")

        if hints["description_text"] and "Description" in fields:
            if _ai_empty(extracted.get("Description")):
                extracted["Description"] = hints["description_text"]

        if hints["website_url"] and "Website URL" in fields:
            ai_url = extracted.get("Website URL", "") or ""
            if _ai_empty(ai_url) or "cloudflare.com" in ai_url:
                extracted["Website URL"] = hints["website_url"]

        if hints["hours"] and "Hours" in fields:
            if _ai_empty(extracted.get("Hours")):
                extracted["Hours"] = hints["hours"]

        if hints["social_links"] and "Social Media Links" in fields:
            if _ai_empty(extracted.get("Social Media Links")):
                extracted["Social Media Links"] = hints["social_links"]

        if hints["gbp_link"] and "GBP Link" in fields:
            if _ai_empty(extracted.get("GBP Link")):
                extracted["GBP Link"] = hints["gbp_link"]

        if hints["category"] and "Category" in fields:
            if _ai_empty(extracted.get("Category")):
                extracted["Category"] = hints["category"]

        # Keywords: prefer hint (already cleaned); clean whatever AI returned
        # Keywords: ONLY from meta keywords tag hint — never from Gemini inference
        if "Keywords" in fields:
            hint_kw = hints.get("keywords", "")
            if hint_kw:
                # Meta keywords tag existed and had content — use cleaned hint (authoritative)
                extracted["Keywords"] = hint_kw
            else:
                # No meta keywords tag on this page — always null, discard Gemini's answer
                extracted["Keywords"] = None

        # Structured-data field overrides for enrollbusiness
        if "enrollbusiness" in source.lower():
            for field_name, hint_key in [
                ("Name", "name"), ("Phone", "phone"), ("Street", "street"),
                ("City", "city"), ("State", "state"), ("Zipcode", "zip"),
                ("Country", "country"),
            ]:
                if hints.get(hint_key) and field_name in fields:
                    if _ai_empty(extracted.get(field_name)):
                        extracted[field_name] = hints[hint_key]

        # FIX v7: enrollbusiness URL-slug fallback when everything is still null
        if "enrollbusiness" in source.lower() and _all_null(extracted, fields):
            slug_data = _parse_enrollbusiness_slug(source)
            if slug_data.get("name") and "Name" in fields:
                extracted["Name"] = slug_data["name"]
            if slug_data.get("city") and "City" in fields:
                extracted["City"] = slug_data["city"]
            if slug_data.get("state") and "State" in fields:
                extracted["State"] = slug_data["state"]
            if slug_data.get("zip") and "Zipcode" in fields:
                extracted["Zipcode"] = slug_data["zip"]
            extracted["_slug_fallback"] = True

        if _all_null(extracted, fields) and (page_html or page_text):
            extracted["_all_null_warning"] = (
                "Gemini returned null for all fields despite page content being present. "
                "Check scrape quality or Cloudflare status."
            )

        # Final Cloudflare URL guard
        if "Website URL" in fields:
            url_val = extracted.get("Website URL", "") or ""
            if "cloudflare.com" in url_val:
                extracted["Website URL"] = None

        # Final hours guard
        if "Hours" in fields:
            hours_val = extracted.get("Hours", "") or ""
            if _PHONE_IN_TEXT.search(hours_val):
                extracted["Hours"] = None
            elif re.search(r"\breview", hours_val, re.IGNORECASE) and not re.search(r"\d{1,2}:\d{2}", hours_val):
                # FIX v7: reject review text masquerading as hours
                extracted["Hours"] = None
            else:
                real_time_segments = len(re.findall(r"\d{1,2}:\d{2}", hours_val))
                placeholder_segments = len(
                    re.findall(r"0{1,2}:0{2}\s*(?:to|-|–)\s*0{1,2}:0{2}", hours_val)
                )
                if real_time_segments > 0 and placeholder_segments == real_time_segments:
                    extracted["Hours"] = None

        # Final keywords guard
        if "Keywords" in fields:
            kw_val = extracted.get("Keywords", "") or ""
            if kw_val:
                extracted["Keywords"] = _clean_keywords(kw_val) or None

        # Final description guard
        if "Description" in fields:
            desc_val = extracted.get("Description", "") or ""
            if desc_val and not _good_desc(desc_val):
                if hints["description_text"] and _good_desc(hints["description_text"]):
                    extracted["Description"] = hints["description_text"]
                else:
                    extracted["Description"] = None

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
            result["_scrape_debug"] = page.get("_debug", "")
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

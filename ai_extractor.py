"""
ai_extractor.py
Sends scraped page content to Google Gemini and extracts
business fields as JSON — including visual fields (Logo, Photos).
No hardcoded layout assumptions. Gemini searches the whole page.

v4 — Bug-fixes over v3:
  - nearfinderus: skip-domains now includes nearfinder.com (catches blog.nearfinder.com)
  - nearfinderus: redirect-wrapper regex broadened to /redirect?url= (not just /empresa/)
  - nearfinderus: description extraction uses broader sibling search + falls back
    to longest paragraph AFTER filtering out pure address strings
  - askmap: description now skips itemprop="description" that looks like an address;
    falls back to the Info/About <p> or longest good paragraph
  - askmap: category extracted via itemprop="category" / breadcrumb
  - enrollbusiness: Cloudflare challenge URL detected → Website URL cleared + page
    treated as blocked at post-process stage
  - All domains: _good_desc min_len raised to 60 to filter address-only strings
  - _good_desc: new address-like pattern filter added
  - nearfinderus/smallbusinessusa/enrollbusiness: category extracted from
    breadcrumb / itemprop / structured data
  - Hours: per-domain patterns tightened; nearfinderus hours extracted from
    the opening-hours table rows directly
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

# FIX: address-like string detector — used to reject false descriptions
_ADDRESS_LIKE = re.compile(
    r"^\s*(address\s*:|phone\s*:|\d+\s+\w+.*\b(blvd|st|ave|rd|ln|dr|way|ct|pl)\b)",
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
    """
    FIX v4: min_len raised from 40→60 to filter address-only strings.
    Also rejects text that looks like a bare address / phone dump.
    """
    if len(text) < min_len:
        return False
    tl = text.lower()
    if any(b in tl for b in _BAD_DESC_PHRASES):
        return False
    # Reject text that is overwhelmingly address/phone data
    if _ADDRESS_LIKE.match(text):
        return False
    # Reject if more than half the tokens look like an address line
    addr_tokens = re.findall(
        r"\b(\d{3,}|blvd|street|avenue|suite|ste|fl\b|zip|phone|tel)\b",
        tl,
    )
    word_count = len(text.split())
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
#  Shared category helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_category_generic(soup) -> str:
    """
    Try several generic strategies to extract the business category:
      1. itemprop="category"
      2. JSON-LD additionalType / knowsAbout
      3. Breadcrumb last item (often the category)
      4. <meta name="keywords"> first token
    """
    # Strategy 1: microdata
    cat = _itemprop(soup, "category")
    if cat and len(cat) < 80:
        return cat

    # Strategy 2: JSON-LD
    ld_blocks = _extract_json_ld(soup)
    for ld in ld_blocks:
        for key in ("additionalType", "knowsAbout", "serviceType", "category"):
            val = ld.get(key, "")
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val and len(val) < 80:
                # Strip schema.org URLs
                val = val.split("/")[-1].replace("-", " ").replace("_", " ")
                return val

    # Strategy 3: breadcrumb — last crumb before current page title
    for nav in soup.find_all(["nav", "ol", "ul", "div"],
                              class_=re.compile(r"breadcrumb", re.I)):
        items = nav.find_all(["li", "a", "span"])
        texts = [i.get_text(strip=True) for i in items if i.get_text(strip=True)]
        if len(texts) >= 2:
            # Second-to-last is usually the category
            candidate = texts[-2]
            if 3 < len(candidate) < 60:
                return candidate

    # Strategy 4: meta keywords first token
    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        kw = meta_kw.get("content", "").split(",")[0].strip()
        if kw and len(kw) < 60:
            return kw

    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Common skip domains
# ─────────────────────────────────────────────────────────────────────────────

# FIX v4: Added nearfinder.com (catches blog.nearfinder.com) and cloudflare.com
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
    """
    FIX v4:
    - itemprop="description" is often just the address on askmap; validate
      with _good_desc before using it.
    - Fall back to the Info <div>/<p> block which holds the real description.
    - Category extracted via _extract_category_generic.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": ""}

    # ── Description ──
    # Priority 1: itemprop="description" — but only if it's a real description
    desc = _itemprop(soup, "description")
    if _good_desc(desc):
        out["description_text"] = desc

    # Priority 2: div/section with class "info", "about", "description"
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

    # Priority 3: heading "Info" / "About" → next sibling
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

    # Priority 4: longest good paragraph
    if not out["description_text"]:
        out["description_text"] = _longest_good_para(soup)

    # ── Category ──
    out["category"] = _extract_category_generic(soup)

    # ── Website ──
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

    # ── Hours ──
    hours_tags = soup.find_all(itemprop="openingHours")
    if hours_tags:
        out["hours"] = "; ".join(t.get_text(strip=True) for t in hours_tags
                                  if t.get_text(strip=True))
    if not out["hours"]:
        spec_tags = soup.find_all(itemprop="openingHoursSpecification")
        if spec_tags:
            out["hours"] = "; ".join(t.get_text(separator=" ", strip=True)
                                      for t in spec_tags if t.get_text(strip=True))

    # ── Social + GBP ──
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
           "social_links": "", "gbp_link": "", "category": ""}

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

    return out


# ── freelistingusa.com ───────────────────────────────────────────────────────

def _extract_freelistingusa(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": ""}

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

    return out


# ── hotfrog.com ───────────────────────────────────────────────────────────────

def _extract_hotfrog(soup) -> dict:
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": ""}

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

        # Category from JSON-LD
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
    FIX v4:
    - Redirect wrapper broadened: match any /redirect?url= pattern (not just /empresa/)
    - nearfinder.com added to _SKIP_DOMAINS so blog.nearfinder.com is filtered
    - Description: "More about" heading search expanded; sibling search now
      also looks at div children; falls back to _longest_good_para
    - Category: extracted from breadcrumb / itemprop
    - Hours: extracted directly from the opening-hours table/list rows
    """
    from urllib.parse import unquote
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": ""}

    # ── Description ──
    # Priority 1: "More about …" section
    for h in soup.find_all(["h2", "h3", "h4", "strong", "b", "p"]):
        htxt = h.get_text(strip=True).lower()
        if re.search(r"more about|about\s+\w", htxt):
            # Check next siblings
            for sib in h.find_next_siblings(["p", "div", "section"]):
                if _is_hidden_tag(sib):
                    continue
                text = sib.get_text(separator=" ", strip=True)
                if _good_desc(text):
                    out["description_text"] = text
                    break
            # Also check direct children of parent
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

    # Priority 2: div with mt-4 class containing a real description paragraph
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

    # Priority 3: any section/div labeled description / about / info
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

    # ── Category ──
    out["category"] = _extract_category_generic(soup)
    # nearfinderus breadcrumb has the category as second-to-last link
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

    # ── Website + Social via redirect wrapper ──
    # FIX v4: broadened regex to catch /redirect?url= anywhere in path
    social_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match any redirect wrapper pattern: /redirect?url=, /empresa/redirect?url=, etc.
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
    # FIX v4: extract directly from the opening-hours table rows
    # nearfinderus renders a table/list with day + time columns
    hours_rows = []
    for container in soup.find_all(["div", "section", "table", "ul"]):
        cls = _cls_str(container)
        if re.search(r"(hours|schedule|working|opening|open)", cls):
            # Table-based hours
            rows = container.find_all("tr")
            if rows:
                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                    if len(cells) >= 2:
                        hours_rows.append(": ".join(cells[:2]))
            if not hours_rows:
                # List / div-row based
                text = container.get_text(separator="|", strip=True)
                if text and len(text) < 600:
                    out["hours"] = text
            if hours_rows:
                out["hours"] = "; ".join(hours_rows)
            if out["hours"]:
                break

    # Fallback: look for day-name patterns in any visible text block
    if not out["hours"]:
        day_pattern = re.compile(
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"mon|tue|wed|thu|fri|sat|sun)",
            re.IGNORECASE,
        )
        for tag in soup.find_all(["p", "div", "li", "span"]):
            text = tag.get_text(separator=" ", strip=True)
            if day_pattern.search(text) and len(text) < 300 and not _is_hidden_tag(tag):
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

def _extract_enrollbusiness(soup) -> dict:
    """
    FIX v4:
    - Category extracted via _extract_category_generic.
    - Hours: "Working Hours:" label used as anchor.
    """
    out = {"description_text": "", "website_url": "", "hours": "",
           "social_links": "", "gbp_link": "", "category": ""}

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

    out["category"] = _extract_category_generic(soup)

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

    # FIX v4: anchor on "Working Hours:" label
    for tag in soup.find_all(["div", "section", "table", "li"]):
        cls = _cls_str(tag)
        text = tag.get_text(separator=" ", strip=True)
        if re.search(r"(working hours|business hours|hours of operation|opening hours)", text, re.I):
            if len(text) < 500:
                # Strip the label prefix and keep the hours portion
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
        # FIX v4: flag when Cloudflare challenge page detected
        "cloudflare_blocked": False,
    }
    if not page_html:
        return hints

    # FIX v4: detect Cloudflare challenge early — no point extracting
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

    except Exception:
        pass

    return hints


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(page_text: str, page_html: str, fields: list, source: str = "") -> str:
    hints = _extract_page_hints(page_html, page_text, source)

    # FIX v4: if Cloudflare blocked, tell Gemini so it doesn't hallucinate
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
    # FIX v4: pass pre-extracted category to AI
    if hints["category"]:
        pre_extracted_facts.append(
            f'CATEGORY (confirmed from page HTML — use as-is):\n{hints["category"]}'
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
                '- "Keywords": tags, keywords, or labels for the business. '
                'Return comma-separated.'
            )
        elif f == "Description":
            field_rules.append(
                '- "Description": the business\'s own descriptive text.\n'
                '  If DESCRIPTION TEXT appears in PRE-EXTRACTED FIELDS above, use it verbatim.\n'
                '  Otherwise find prose paragraphs describing what the business does.\n'
                '  EXCLUDE: navigation text, review prompts, "Payment methods", '
                '"Last update", footer text, bare address strings, phone numbers.\n'
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
- For Website URL: NEVER return a cloudflare.com URL — return null if only cloudflare URLs exist.
- For Description: NEVER return a bare address string or phone number as the description.

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

        # ── POST-PROCESS: authoritative overrides ─────────────────────────
        hints = _extract_page_hints(page_html, page_text, source)

        # FIX v4: if Cloudflare challenge, null all fields
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

        # Only override text fields if AI returned null/empty AND we have a value
        if hints["description_text"] and "Description" in fields:
            if not extracted.get("Description"):
                extracted["Description"] = hints["description_text"]
        if hints["website_url"] and "Website URL" in fields:
            ai_url = extracted.get("Website URL", "") or ""
            # FIX v4: also clear Cloudflare URLs that Gemini may have returned
            if not ai_url or "cloudflare.com" in ai_url:
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
        # FIX v4: category override
        if hints["category"] and "Category" in fields:
            if not extracted.get("Category"):
                extracted["Category"] = hints["category"]

        # FIX v4: final Cloudflare URL guard regardless of source
        if "Website URL" in fields:
            url_val = extracted.get("Website URL", "") or ""
            if "cloudflare.com" in url_val:
                extracted["Website URL"] = None

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

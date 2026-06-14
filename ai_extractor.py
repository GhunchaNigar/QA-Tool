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
#  Shared constants
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

_UI_IMAGE_PATTERNS = re.compile(
    r"(icon|sprite|arrow|chevron|star-rating|rating-star|badge|flag|"
    r"banner-ad|advertisement|ad[-_]|[-_]ad\.|pixel\.gif|blank\.gif|"
    r"spacer|placeholder|loading|spinner|ajax-loader|no[-_]?image|"
    r"default[-_]avatar|generic[-_])",
    re.IGNORECASE,
)

_LOGO_SIGNALS = re.compile(
    r"(logo|brand|profile[-_]?img|profile[-_]?pic|avatar|"
    r"business[-_]?img|company[-_]?img|listing[-_]?img|thumb)",
    re.IGNORECASE,
)

_PHOTO_SIGNALS = re.compile(
    r"(photo|gallery|image|images|media|cover|banner|hero|"
    r"carousel|slide|slider|backdrop|background|uploads)",
    re.IGNORECASE,
)

_ADDRESS_LIKE = re.compile(
    r"^\s*(address\s*:|phone\s*:|\d+\s+\w+.*\b(blvd|st|ave|rd|ln|dr|way|ct|pl)\b)",
    re.IGNORECASE,
)

_ROAD_TYPES = re.compile(
    r"\b(blvd|boulevard|street|st\b|avenue|ave\b|road\b|rd\b|lane\b|ln\b|"
    r"drive\b|dr\b|way\b|court\b|ct\b|place\b|pl\b|circle\b|cir\b|"
    r"parkway\b|pkwy\b|highway\b|hwy\b|suite\b|ste\b|floor\b|fl\b)\b",
    re.IGNORECASE,
)

_PHONE_IN_TEXT = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")

_BAD_DESC_PHRASES = (
    "payment methods", "last update", "general information",
    "write a review", "sign up", "site map", "privacy policy",
    "terms of", "cookie", "copyright", "all rights reserved",
    "opening hours", "phone number", "get directions",
    "claim this", "report an error", "edit this",
    "add to favorites", "share this", "follow us",
)

# Regex to detect domain-like names (e.g. "nearfinderus.com")
_DOMAIN_NAME_RE = re.compile(
    r"\.(com|net|org|io|co|us|info|biz|gov|edu)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cls_str(tag) -> str:
    return (" ".join(tag.get("class", [])) + " " + tag.get("id", "")).lower()


def _all_srcs(tag) -> list:
    attrs = ("src", "data-src", "data-lazy", "data-lazy-src",
             "data-original", "data-url", "data-image", "data-bg",
             "data-background", "data-srcset")
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


def _is_tiny(img, threshold: int = 50) -> bool:
    try:
        w = int(img.get("width", 0) or 0)
        h = int(img.get("height", 0) or 0)
        if w > 0 and w < threshold:
            return True
        if h > 0 and h < threshold:
            return True
    except (ValueError, TypeError):
        pass
    return False


def _is_hidden(tag) -> bool:
    for el in [tag] + list(tag.parents):
        if not hasattr(el, "get"):
            continue
        style = el.get("style", "").lower()
        if "display: none" in style or "display:none" in style:
            return True
        if "visibility: hidden" in style or "visibility:hidden" in style:
            return True
    return False


# ── Site-chrome / ad / directory-brand filters (for Logo & Photos detection) ──

_CHROME_TAGS = {"header", "footer", "nav", "aside"}

_CHROME_OR_AD_RE = re.compile(
    r"\b(header|footer|nav(bar)?|topbar|masthead|sidebar|"
    r"promo|sponsor(ed)?|advert(isement)?s?|"
    r"banner-ad|google-ad|adsbygoogle|ad[-_]slot|ad[-_]unit|"
    r"site-logo|navbar-brand|brand-logo|"
    r"search-widget|custom-search)\b",
    re.IGNORECASE,
)

_AD_NETWORK_RE = re.compile(
    r"(doubleclick|googlesyndication|googleadservices|adsystem|adnxs|"
    r"taboola|outbrain|criteo|tiktok|pagead|gstatic\.com|"
    r"google\.com/(?:logos|images)|trustrank)",
    re.IGNORECASE,
)


def _is_chrome_or_ad(tag) -> bool:
    """True if tag (or an ancestor) looks like site chrome, navigation, or an ad/widget —
    i.e. NOT part of the actual business listing content."""
    for el in [tag] + list(tag.parents):
        if not hasattr(el, "name"):
            continue
        if el.name in _CHROME_TAGS:
            return True
        if hasattr(el, "get") and _CHROME_OR_AD_RE.search(_cls_str(el)):
            return True
    return False


def _directory_brand(source: str) -> str:
    """
    Best-effort extraction of the directory site's own brand/short-name from
    its domain, used to filter out the SITE's logo (e.g. 'gravitysplash' for
    gravitysplash.com) from business-logo detection.
    """
    if not source:
        return ""
    labels = [p for p in source.lower().split(".") if p]
    skip = {"www", "us", "app", "en", "m", "en-us"}
    for label in labels:
        if label in skip:
            continue
        if len(label) > 2:
            return label
    return labels[0] if labels else ""


def _good_desc(text: str, min_len: int = 60) -> bool:
    if not text or len(text) < min_len:
        return False
    tl = text.lower()
    if any(b in tl for b in _BAD_DESC_PHRASES):
        return False
    if _ADDRESS_LIKE.match(text):
        return False
    words = re.sub(r"[–—]", " ", text).split()
    if len(words) < 12 and _ROAD_TYPES.search(text):
        return False
    if re.search(r"\b\d{5}\b", text):
        return False
    addr_tokens = re.findall(
        r"\b(\d{3,}|blvd|street|avenue|suite|ste|fl\b|zip|phone|tel)\b",
        text.lower(),
    )
    if words and len(addr_tokens) / len(words) > 0.35:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Structured data extractors (JSON-LD, microdata, meta)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_ld(soup) -> list:
    results = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = (script.string or "").strip()
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


def _ld_find(blocks: list, *types_) -> dict:
    type_lower = [t.lower() for t in types_]
    for block in blocks:
        bt = block.get("@type", "")
        if isinstance(bt, list):
            bt = " ".join(bt)
        if any(t in bt.lower() for t in type_lower):
            return block
    for block in blocks:
        if block.get("@type"):
            return block
    return {}


def _itemprop(soup, prop: str, attr: str = "text") -> str:
    tag = soup.find(itemprop=prop)
    if tag is None:
        return ""
    if attr == "text":
        return tag.get_text(strip=True)
    return tag.get(attr, "").strip()


def _itemprop_all(soup, prop: str) -> list:
    return [t.get_text(strip=True) for t in soup.find_all(itemprop=prop)]


def _ld_address(ld: dict) -> dict:
    addr = ld.get("address", {})
    if isinstance(addr, str):
        return {}
    if not isinstance(addr, dict):
        return {}
    return {
        "street":  addr.get("streetAddress", ""),
        "city":    addr.get("addressLocality", ""),
        "state":   addr.get("addressRegion", ""),
        "zip":     addr.get("postalCode", ""),
        "country": addr.get("addressCountry", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Universal structured data extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_structured(soup) -> dict:
    """
    Pull every field from JSON-LD and microdata (schema.org).
    These standards are used by all major directories.
    Returns a flat dict of raw hint values (may be empty strings).
    """
    out = {
        "name": "", "phone": "", "street": "", "city": "", "state": "",
        "zip": "", "country": "", "description": "", "website": "",
        "hours": "", "category": "", "email": "",
        "logo_url": "", "image_urls": [],
        "social_links": [],
    }

    ld_blocks = _extract_json_ld(soup)
    ld = _ld_find(
        ld_blocks,
        "LocalBusiness", "Organization", "Store", "Restaurant",
        "Service", "ProfessionalService", "HomeAndConstructionBusiness",
        "MedicalBusiness", "HealthAndBeautyBusiness", "LegalService",
    )

    # ── JSON-LD ──
    if ld:
        out["name"]  = ld.get("name", "")
        out["phone"] = ld.get("telephone", ld.get("phone", ""))
        out["email"] = ld.get("email", "")

        addr = _ld_address(ld)
        out["street"]  = addr.get("street", "")
        out["city"]    = addr.get("city", "")
        out["state"]   = addr.get("state", "")
        out["zip"]     = addr.get("zip", "")
        out["country"] = addr.get("country", "")

        desc = ld.get("description", "")
        if _good_desc(desc):
            out["description"] = desc

        url = ld.get("url", "")
        if url and not any(s in url for s in _SKIP_DOMAINS):
            out["website"] = url

        # Hours
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

        # Category
        for key in ("additionalType", "knowsAbout", "serviceType", "category"):
            val = ld.get(key, "")
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val:
                out["category"] = val.split("/")[-1].replace("-", " ").replace("_", " ")
                break

        # Logo
        logo_val = ld.get("logo", "")
        if isinstance(logo_val, dict):
            logo_val = logo_val.get("url", logo_val.get("contentUrl", ""))
        if isinstance(logo_val, str) and logo_val.startswith("http"):
            out["logo_url"] = logo_val

        # Images
        images = ld.get("image", [])
        if isinstance(images, str) and images.startswith("http"):
            out["image_urls"].append(images)
        elif isinstance(images, list):
            out["image_urls"].extend(i for i in images if isinstance(i, str) and i.startswith("http"))

        # Social (sameAs)
        same_as = ld.get("sameAs", [])
        if isinstance(same_as, str):
            same_as = [same_as]
        for link in same_as:
            if any(s in link for s in _SOCIAL_DOMAINS):
                out["social_links"].append(link)

    # ── Microdata (itemprop) — fills gaps left by JSON-LD ──
    if not out["name"]:
        out["name"] = _itemprop(soup, "name")
    if not out["phone"]:
        out["phone"] = _itemprop(soup, "telephone")
    if not out["email"]:
        out["email"] = _itemprop(soup, "email")
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
    if not out["description"]:
        desc = _itemprop(soup, "description")
        if _good_desc(desc):
            out["description"] = desc
    if not out["hours"]:
        oh_tags = soup.find_all(attrs={"itemprop": "openingHours"})
        if oh_tags:
            hrs = []
            for t in oh_tags:
                val = t.get("content", "").strip() or t.get_text(strip=True)
                if val and not re.search(r"0{1,2}:0{2}\s*(?:to|-|–)\s*0{1,2}:0{2}", val):
                    hrs.append(val)
            if hrs:
                out["hours"] = "; ".join(hrs)

    # ── Meta tags ──
    og_img = (soup.find("meta", property="og:image") or
               soup.find("meta", attrs={"name": "og:image"}))
    if og_img:
        content = og_img.get("content", "")
        if content.startswith("http") and not _UI_IMAGE_PATTERNS.search(content):
            if not out["logo_url"]:
                out["logo_url"] = content
            if content not in out["image_urls"]:
                out["image_urls"].append(content)

    if not out["description"]:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            content = meta_desc.get("content", "").strip()
            if _good_desc(content):
                out["description"] = content

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Universal DOM heuristics (no domain assumptions)
# ─────────────────────────────────────────────────────────────────────────────

def _longest_good_para(soup) -> str:
    best = ""
    for p in soup.find_all("p"):
        if _is_hidden(p):
            continue
        text = p.get_text(separator=" ", strip=True)
        if _good_desc(text) and len(text) > len(best):
            best = text
    return best


_FALSE_DESC_RE = re.compile(
    r"^[^.]{0,80}Category\s*:\s*[^.]{0,80}$"
    r"|^\s*\w[\w\s&,\-\.]+\s+Category\s*:\s*[\w\s&,\-]+$",
    re.IGNORECASE,
)

_NAV_CONTAINER_RE = re.compile(
    r"\b(breadcrumb|nav|menu|footer|header|sidebar|widget|"
    r"related|recommend|sponsored|ad[-_]|[-_]ad\b)\b",
    re.IGNORECASE,
)


def _is_false_description(text: str) -> bool:
    if not text:
        return True
    if re.search(r"\bCategory\s*:", text, re.IGNORECASE) and len(text) < 200:
        return True
    if text.count("|") >= 2 and len(text) < 200:
        return True
    word_count = len(text.split())
    has_sentence = bool(re.search(r"[.!?]", text))
    if word_count < 12 and not has_sentence:
        return True
    return False


def _container_is_nav(tag) -> bool:
    cls = _cls_str(tag)
    if _NAV_CONTAINER_RE.search(cls):
        return True
    parent = tag.parent
    if parent and hasattr(parent, "get"):
        if _NAV_CONTAINER_RE.search(_cls_str(parent)):
            return True
    return False


def _universal_description(soup, structured_desc: str) -> str:
    def _better(a: str, b: str) -> str:
        a_ok = _good_desc(a) and not _is_false_description(a)
        b_ok = _good_desc(b) and not _is_false_description(b)
        if a_ok and b_ok:
            return b if len(b) >= len(a) else a
        if b_ok:
            return b
        if a_ok:
            return a
        return ""

    best = ""

    if structured_desc and _good_desc(structured_desc) and not _is_false_description(structured_desc):
        if len(structured_desc) >= 150:
            best = structured_desc

    more_about_re = re.compile(r"\bmore\s+about\b", re.IGNORECASE)
    for h in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
        if not more_about_re.search(h.get_text(strip=True)):
            continue
        container = h.parent
        if container:
            combined = " ".join(
                p.get_text(separator=" ", strip=True)
                for p in container.find_all("p")
                if not _is_hidden(p)
            ).strip()
            if combined and len(combined) > 50:
                best = _better(best, combined)
                if len(best) >= 200:
                    break
        for sib in h.find_next_siblings(["p", "div", "section"]):
            if _is_hidden(sib):
                continue
            text = sib.get_text(separator=" ", strip=True)
            if _good_desc(text) and not _is_false_description(text):
                best = _better(best, text)
                break

    if not best or len(best) < 150:
        about_re = re.compile(
            r"\b(about(\s+us)?|overview|description|who\s+we\s+are|our\s+story|"
            r"company\s+info|business\s+info)\b",
            re.IGNORECASE,
        )
        for h in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
            htxt = h.get_text(strip=True)
            if not about_re.search(htxt):
                continue
            if _container_is_nav(h):
                continue
            for sib in h.find_next_siblings(["p", "div", "section"]):
                if _is_hidden(sib):
                    continue
                text = sib.get_text(separator=" ", strip=True)
                if _good_desc(text) and not _is_false_description(text):
                    best = _better(best, text)
                    break

    if not best or len(best) < 150:
        desc_cls_re = re.compile(
            r"\b(description|about|overview|summary|bio|info[-_]?text|"
            r"company[-_]?info|business[-_]?info|listing[-_]?desc)\b",
            re.IGNORECASE,
        )
        for tag in soup.find_all(["div", "section", "article", "p"]):
            if _is_hidden(tag) or _container_is_nav(tag):
                continue
            if not desc_cls_re.search(_cls_str(tag)):
                continue
            text = tag.get_text(separator=" ", strip=True)
            if _good_desc(text) and not _is_false_description(text):
                best = _better(best, text)

    if not best or len(best) < 150:
        all_paras = soup.find_all("p")
        for p in reversed(all_paras):
            if _is_hidden(p) or _container_is_nav(p):
                continue
            text = p.get_text(separator=" ", strip=True)
            if _good_desc(text) and not _is_false_description(text):
                best = _better(best, text)
                if len(best) >= 300:
                    break

    if not best:
        if structured_desc and _good_desc(structured_desc):
            return structured_desc
        return _longest_good_para(soup)

    return best


def _universal_website(soup) -> str:
    website_re = re.compile(r"(visit website|official site|our website|web site|homepage)", re.IGNORECASE)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        txt = a.get_text(strip=True)
        cls = _cls_str(a)
        if not href.startswith("http"):
            continue
        if any(s in href for s in _SKIP_DOMAINS):
            continue
        if website_re.search(txt) or "website" in cls or "external" in cls:
            return href

    url_tag = soup.find(itemprop="url")
    if url_tag:
        href = url_tag.get("href", "") or url_tag.get("content", "")
        if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
            return href

    for a in soup.find_all("a", href=True, rel=True):
        rel = " ".join(a.get("rel", []))
        href = a["href"]
        if ("nofollow" in rel or "external" in rel) and href.startswith("http"):
            if not any(s in href for s in _SKIP_DOMAINS):
                return href

    from urllib.parse import unquote
    for a in soup.find_all("a", href=True):
        href = a["href"]
        redirect_match = re.search(r"/redirect\?url=([^&\s]+)", href)
        if redirect_match:
            decoded = unquote(redirect_match.group(1))
            if not any(s in decoded for s in _SKIP_DOMAINS) and not any(s in decoded for s in _SOCIAL_DOMAINS):
                return decoded

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and not any(s in href for s in _SKIP_DOMAINS):
            return href

    return ""


def _universal_social(soup) -> str:
    from urllib.parse import unquote
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(s in href for s in _SOCIAL_DOMAINS):
            links.append(href)
            continue
        redirect_match = re.search(r"/redirect\?url=([^&\s]+)", href)
        if redirect_match:
            decoded = unquote(redirect_match.group(1))
            if any(s in decoded for s in _SOCIAL_DOMAINS):
                links.append(decoded)

    return ", ".join(dict.fromkeys(links))


def _universal_hours(soup, structured_hours: str) -> str:
    _PLACEHOLDER_RE = re.compile(
        r"0{1,2}:0{2}\s*(?:to|-|–|–)\s*0{1,2}:0{2}", re.IGNORECASE
    )
    _DAY_RE = re.compile(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"mon|tue|wed|thu|fri|sat|sun)\b",
        re.IGNORECASE,
    )
    _TIME_RE = re.compile(
        r"\d{1,2}:\d{2}|\b(am|pm|open|closed|24\s*hours?)\b", re.IGNORECASE
    )
    _HOURS_CONTAINER_RE = re.compile(
        r"\b(hours|schedule|working[-_]?hours|business[-_]?hours|"
        r"opening[-_]?hours|open[-_]?time|timetable)\b",
        re.IGNORECASE,
    )

    def _is_placeholder_only(text: str) -> bool:
        real_times = re.findall(r"\d{1,2}:\d{2}", text)
        if not real_times:
            return False
        return all(re.match(r"^0{1,2}:0{2}$", t) for t in real_times)

    def _hours_valid(text: str) -> bool:
        if not text or len(text) > 700:
            return False
        if _PHONE_IN_TEXT.search(text):
            return False
        if re.search(r"\breview\b", text, re.IGNORECASE) and not _TIME_RE.search(text):
            return False
        if _is_placeholder_only(text):
            return False
        placeholder_count = len(_PLACEHOLDER_RE.findall(text))
        day_count = len(_DAY_RE.findall(text))
        if day_count > 0 and placeholder_count >= day_count:
            return False
        return bool(_DAY_RE.search(text) and _TIME_RE.search(text))

    if structured_hours and not _is_placeholder_only(structured_hours):
        if _hours_valid(structured_hours) or re.search(r"\b(open|closed|24)\b", structured_hours, re.IGNORECASE):
            return structured_hours

    oh_tags = soup.find_all(attrs={"itemprop": "openingHours"})
    if oh_tags:
        hrs = []
        for t in oh_tags:
            val = (t.get("content", "") or t.get_text(strip=True)).strip()
            if val and not _is_placeholder_only(val):
                hrs.append(val)
        if hrs:
            candidate = "; ".join(hrs)
            if not _is_placeholder_only(candidate):
                return candidate

    for table in soup.find_all("table"):
        cls = _cls_str(table)
        parent_cls = _cls_str(table.parent) if table.parent and hasattr(table.parent, "get") else ""
        if not (_HOURS_CONTAINER_RE.search(cls) or _HOURS_CONTAINER_RE.search(parent_cls)):
            continue
        rows = table.find_all("tr")
        if not rows:
            continue
        row_parts = []
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            day_part  = cells[0].strip()
            time_part = cells[1].strip()
            if not _DAY_RE.search(day_part):
                continue
            if _is_placeholder_only(time_part) or not time_part:
                continue
            row_parts.append(f"{day_part}: {time_part}")
        if row_parts:
            return "; ".join(row_parts)

    for tag in soup.find_all(["div", "section", "ul", "dl"]):
        cls = _cls_str(tag)
        if not _HOURS_CONTAINER_RE.search(cls):
            continue
        if re.search(r"\b(address|contact|phone|map)\b", cls):
            continue
        if _is_hidden(tag):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if _hours_valid(text):
            return text

    for tag in soup.find_all(["p", "div", "li", "span", "td", "dd"]):
        if _is_hidden(tag):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if _hours_valid(text) and len(text) < 400:
            return text

    return ""


def _universal_category(soup) -> str:
    cat = _itemprop(soup, "category")
    if cat and len(cat) < 80:
        return cat

    for nav in soup.find_all(["nav", "ol", "ul", "div"],
                              class_=re.compile(r"breadcrumb", re.I)):
        items = nav.find_all(["li", "a", "span"])
        texts = [i.get_text(strip=True) for i in items if i.get_text(strip=True)]
        if len(texts) >= 2:
            candidate = texts[-2]
            if 3 < len(candidate) < 60:
                return candidate

    return ""


"""
PATCH: Replace the entire _universal_keywords() function in ai_extractor.py
with this version.

Changes vs original
───────────────────
1. Strategy A — "Business tags" heading → siblings only (not parent)
   The original collected from the heading's PARENT container, which on brownbook
   also contains the "Location tags" column.  We now walk only NEXT SIBLINGS of
   the heading, stopping before any location-tag element.  The parent-container
   fallback is kept but now skips descendants that live inside a location
   sub-container (exclude_location_subtrees=True).

2. _collect_tag_items() — exclude location sub-trees by default
   New parameter `exclude_location_subtrees` (default True).  When set, any
   child element whose ancestor chain (up to `container`) passes through a
   location-tag container is silently skipped.  This is the surgical fix that
   stops "Nokomis, Florida" leaking into business-tag results on brownbook.

3. Strategy C — plain-text colon AND next-line patterns
   Handles "Tags: Water Damage Restoration Service" inline (freelistingusa) and
   the two-line "Tags\nWater Damage Restoration Service" FAQ-block variant.
   Previously only the next-line regex was tried, and it required a blank line
   between the label and the value.

4. Strategy D — relaxed label matching
   The original regex `^(tags?)$` required the label element to contain
   ONLY the word "Tags".  The new version also matches when the element text
   STARTS WITH "Tags" (e.g. "Tags:" with trailing colon inside the node),
   catching more real-world patterns.
"""

def _universal_keywords(
    soup,
    business_name: str = "",
    category: str = "",
    city: str = "",
    state: str = "",
) -> str:  # noqa: C901
    """
    Keywords from:
      1. <meta name="keywords"> — highest priority, used verbatim
      2. "Business tags" / "Tags" DOM sections — explicitly excluding "Location tags"

    KEY RULES:
    - Never split multi-word tags on spaces (preserves "ChatGPT Ads Agency" intact)
    - "Location tags" sibling containers and sub-trees are excluded
    - Only comma, pipe, semicolon, or period used as tag separators
    - business_name / category / city / state are passed through to
      _clean_keywords() so that <meta name="keywords"> blobs (which often
      mash together the business name + category + city) don't leak those
      values into the final Keywords field.
    """

    # ── Shared patterns ────────────────────────────────────────────────────────
    _BIZ_TAG_RE = re.compile(
        r"\b(business\s+tags?|tags?|keywords?|services?\s+offered)\b",
        re.IGNORECASE,
    )
    _LOC_TAG_RE = re.compile(
        r"\b(location\s+tags?|location|city|cities|region|area|"
        r"local\s+tags?|geo\s+tags?)\b",
        re.IGNORECASE,
    )

    def _is_location_container(tag) -> bool:
        cls = _cls_str(tag)
        txt = tag.get_text(strip=True)[:80]
        return bool(_LOC_TAG_RE.search(cls) or _LOC_TAG_RE.search(txt))

    def _collect_tag_items(container, exclude_location_subtrees: bool = True) -> list:
        """
        Collect individual keyword/tag texts from a container.

        exclude_location_subtrees=True (default): skip any child element whose
        ancestor chain (up to `container`) passes through a location-tag node.
        This prevents brownbook's "Location tags" column from contaminating
        results when both columns share a parent wrapper.

        Prefers <a>/<span>/<li> children (each element = one keyword).
        Falls back to comma/pipe/semicolon splitting of plain text.
        Never splits on spaces so multi-word phrases stay intact.
        """
        items = []
        children = container.find_all(
            ["a", "span", "li", "strong", "em"], recursive=True
        )
        for child in children:
            if exclude_location_subtrees:
                in_loc = False
                for ancestor in child.parents:
                    if ancestor is container:
                        break
                    if hasattr(ancestor, "get") and _is_location_container(ancestor):
                        in_loc = True
                        break
                if in_loc:
                    continue

            txt = child.get_text(strip=True)

            if child.name == "a":
                href = child.get("href", "")
                if href and not href.startswith("#"):
                    if any(d in href for d in _SKIP_DOMAINS):
                        continue

            if txt and 2 < len(txt) < 100:
                items.append(txt)

        if items:
            return items

        # Fallback: split plain text on separators only (never spaces)
        raw = container.get_text(strip=True)
        parts = re.split(r"[,|;.]", raw)
        return [p.strip() for p in parts if p.strip() and 2 < len(p.strip()) < 100]

    # ── Priority 1: <meta name="keywords"> ────────────────────────────────────
    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        raw = meta_kw.get("content", "").strip()
        if raw:
            cleaned_meta = _clean_keywords(
                raw, business_name=business_name, category=category, city=city, state=state
            )
            if cleaned_meta:
                # If the meta keywords blob reduces to a single token that is
                # just the Category restated (e.g. <meta name="keywords"
                # content="Water Damage Restoration"> on a page whose only
                # other label is "Category: Water Damage Restoration"), don't
                # treat that as a real keywords/tags list — fall through to
                # the structural strategies below (which may find a genuine
                # "Business tags" section, or find nothing → null).
                single_tokens = [t.strip() for t in cleaned_meta.split(",") if t.strip()]
                cat_norm = category.lower().strip()
                if not (len(single_tokens) == 1 and cat_norm
                        and single_tokens[0].lower().strip() == cat_norm):
                    return cleaned_meta

    # ── Strategy A: "Business tags" heading → walk NEXT siblings only ─────────
    #
    # brownbook layout (simplified):
    #   <div class="tags-wrapper">
    #     <div class="col">
    #       <h3>Business tags</h3>          ← we find this
    #       <a>Water Damage …</a>
    #     </div>
    #     <div class="col">
    #       <h3>Location tags</h3>          ← we STOP before this
    #       <a>Nokomis</a><a>Florida</a>
    #     </div>
    #   </div>
    #
    # Fix: walk NEXT SIBLINGS of the heading (not its parent's all-children).
    # Parent-container fallback kept but now skips location sub-trees.

    for label_tag in soup.find_all(
        ["h2", "h3", "h4", "h5", "strong", "b", "p", "div", "span", "td", "th"]
    ):
        label_txt = label_tag.get_text(strip=True)

        if not _BIZ_TAG_RE.search(label_txt):
            continue
        if _LOC_TAG_RE.search(label_txt):          # skip "Location tags" headings
            continue
        if _container_is_nav(label_tag):
            continue

        # --- Walk NEXT siblings of the heading (primary path) -----------------
        collected = []
        for sib in label_tag.find_next_siblings():
            if not hasattr(sib, "get_text"):
                continue
            sib_txt = sib.get_text(strip=True)
            if not sib_txt:
                continue
            if _is_location_container(sib):        # stop at location section
                break
            if hasattr(sib, "name") and sib.name in ("h2", "h3", "h4"):
                if not _BIZ_TAG_RE.search(sib_txt):
                    break
            items = _collect_tag_items(sib, exclude_location_subtrees=True)
            if items:
                collected.extend(items)
                break  # consume only the first non-empty content sibling

        if collected:
            result = _clean_keywords(", ".join(collected), business_name=business_name, category=category, city=city, state=state)
            if result:
                return result

        # --- Parent-container fallback (skip location sub-trees) --------------
        parent = label_tag.parent
        if parent and hasattr(parent, "get"):
            if not _is_location_container(parent):
                items = _collect_tag_items(parent, exclude_location_subtrees=True)
                items = [i for i in items if i.lower() != label_txt.lower()]
                if items:
                    result = _clean_keywords(", ".join(items), business_name=business_name, category=category, city=city, state=state)
                    if result:
                        return result

    # ── Strategy B: containers with tag/keyword class/id ──────────────────────
    _TAG_CLS_RE = re.compile(
        r"\b(business[-_]?tags?|listing[-_]?tags?|tags?[-_]?list|"
        r"keywords?[-_]?list|tag[-_]?container|tag[-_]?cloud|chips?|"
        r"tag[-_]?wrapper|tag[-_]?group)\b",
        re.IGNORECASE,
    )
    for container in soup.find_all(["div", "ul", "section", "p", "span"]):
        cls = _cls_str(container)
        if not _TAG_CLS_RE.search(cls):
            continue
        if _is_location_container(container):
            continue
        if _is_hidden(container):
            continue
        items = _collect_tag_items(container, exclude_location_subtrees=True)
        if items:
            result = _clean_keywords(", ".join(items), business_name=business_name, category=category, city=city, state=state)
            if result:
                return result

    # ── Strategy C: plain-text colon and next-line patterns ───────────────────
    #
    # Handles freelistingusa "Tags: Water Damage Restoration Service" (same line)
    # and two-line FAQ-block "Tags\nWater Damage Restoration Service".

    full_text = soup.get_text(separator="\n", strip=True)

    # Same-line colon: "Tags: foo, bar"  or  "Business tags: foo"
    for pattern in [
        r"(?:^|[\n\r])\s*(?:business\s+)?tags?\s*:\s*([^\n\r]{3,300})",
        r"(?:^|[\n\r])\s*keywords?\s*:\s*([^\n\r]{3,300})",
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
        if m:
            raw = m.group(1).strip()
            parts = [p.strip() for p in re.split(r"[,|;.]", raw) if p.strip()]
            cleaned = _clean_keywords(", ".join(parts), business_name=business_name, category=category, city=city, state=state)
            if cleaned:
                return cleaned

    # Next-line: label on line N, values on line N+1
    for pattern in [
        r"(?:^|[\n\r])\s*(?:business\s+)?tags?\s*[\n\r]+\s*([^\n\r]{3,300})",
        r"(?:^|[\n\r])\s*keywords?\s*[\n\r]+\s*([^\n\r]{3,300})",
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
        if m:
            raw = m.group(1).strip()
            if len(raw.split()) > 20:      # skip if it looks like a paragraph
                continue
            parts = [p.strip() for p in re.split(r"[,|;.]", raw) if p.strip()]
            cleaned = _clean_keywords(", ".join(parts), business_name=business_name, category=category, city=city, state=state)
            if cleaned:
                return cleaned

    # ── Strategy D: DOM label element → next sibling (relaxed matching) ───────
    #
    # Original required element text == exactly "Tags" (strict `^(tags?)$`).
    # New: also matches when element text STARTS WITH "Tags" (e.g. "Tags:").

    _LABEL_STRICT_RE = re.compile(
        r"^(tags?|business\s+tags?|keywords?)\s*:?\s*$",
        re.IGNORECASE,
    )
    _LABEL_STARTS_RE = re.compile(
        r"^(tags?|business\s+tags?|keywords?)\b",
        re.IGNORECASE,
    )

    for tag in soup.find_all(["h2", "h3", "h4", "strong", "b", "p", "div", "span"]):
        txt = tag.get_text(strip=True)
        if not (_LABEL_STRICT_RE.match(txt) or _LABEL_STARTS_RE.match(txt)):
            continue
        if _LOC_TAG_RE.search(txt):
            continue
        if _container_is_nav(tag):
            continue
        for sib in tag.find_next_siblings():
            sib_txt = sib.get_text(strip=True) if hasattr(sib, "get_text") else ""
            if not sib_txt or _is_location_container(sib):
                continue
            if len(sib_txt) < 300:
                items = _collect_tag_items(sib, exclude_location_subtrees=True)
                if not items:
                    items = [p.strip() for p in re.split(r"[,|;.]", sib_txt)
                             if p.strip()]
                if items:
                    cleaned = _clean_keywords(", ".join(items), business_name=business_name, category=category, city=city, state=state)
                    if cleaned:
                        return cleaned
            break

    return ""


def _universal_gbp(soup) -> str:
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "google.com" in href and any(p in href for p in ("maps/place", "maps?q", "goo.gl")):
            return href
    return ""


def _universal_email(soup) -> str:
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            return href[7:].split("?")[0].strip()
    text = soup.get_text()
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    if match:
        return match.group(0)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Universal visual detectors (logo / photos)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_logo(soup, structured: dict, source: str = "") -> bool:
    if structured.get("logo_url"):
        return True

    brand = _directory_brand(source)

    tag = soup.find(attrs={"itemprop": "image"})
    if tag:
        src = tag.get("src", tag.get("content", ""))
        if src and src.startswith("http") and not _is_tiny(tag, 20):
            return True

    for img in soup.find_all("img"):
        srcs = _all_srcs(img)
        if not srcs:
            continue
        src = srcs[0]
        cls = _cls_str(img)
        parent_cls = " ".join(
            _cls_str(p) for p in img.parents
            if hasattr(p, "get") and p.name not in ("html", "body")
        )[:300]
        alt = img.get("alt", "").lower()

        if _is_tiny(img, 20) or _UI_IMAGE_PATTERNS.search(src):
            continue

        # Skip the directory site's own logo/branding, ads, and nav/header/footer chrome
        if _is_chrome_or_ad(img):
            continue
        if _AD_NETWORK_RE.search(src) or _AD_NETWORK_RE.search(parent_cls):
            continue
        if brand and (brand in src.lower() or brand in cls
                       or brand in parent_cls.lower() or brand in alt):
            continue

        if (_LOGO_SIGNALS.search(src) or _LOGO_SIGNALS.search(cls)
                or _LOGO_SIGNALS.search(parent_cls)
                or re.search(r"\b(logo|brand|emblem)\b", alt)):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Reference-logo matching (hardcoded business logo → exact-match detection)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_logo_candidate_urls(soup, structured: dict, source: str = "") -> list:
    """
    Collect candidate image URLs that MIGHT be the business's logo, in
    priority order. These are then checked against the user's reference
    logo via perceptual hashing (logo_matcher) — only an actual match
    counts, so it's fine (and better) to be generous here.
    """
    candidates = []

    def _add(url):
        if url and url not in candidates:
            candidates.append(url)

    # Highest priority: structured data (JSON-LD logo, og:image, itemprop image)
    if structured.get("logo_url"):
        _add(structured["logo_url"])
    for url in structured.get("image_urls", []):
        _add(url)

    tag = soup.find(attrs={"itemprop": "image"})
    if tag:
        _add(tag.get("src", tag.get("content", "")))

    # Collect ALL <img> elements — every src variant (src, data-src, srcset, etc.)
    # is included so no image is ever skipped.  The only filter applied here is
    # the ad-network RE, which blocks known tracking pixels and ad-serving URLs
    # that would never be a business logo.  All other filtering (tiny size,
    # site-chrome, directory branding) is intentionally removed because the
    # perceptual-hash comparison against the reference logo is the real filter:
    # an image that doesn't match the reference simply doesn't contribute to the
    # result, so being maximally inclusive here costs only a few extra downloads
    # while preventing false-negative misses when the logo appears in an
    # unexpected DOM location.
    for img in soup.find_all("img"):
        all_img_srcs = _all_srcs(img)
        if not all_img_srcs:
            continue
        parent_cls = " ".join(
            _cls_str(p) for p in img.parents
            if hasattr(p, "get") and p.name not in ("html", "body")
        )[:300]
        for src in all_img_srcs:
            if _AD_NETWORK_RE.search(src) or _AD_NETWORK_RE.search(parent_cls):
                continue
            _add(src)

    # Also collect CSS background-image URLs — some directories display the
    # business logo as a background-image on a <div> rather than an <img>.
    import re as _re
    for tag in soup.find_all(style=True):
        bg_urls = _re.findall(
            r"background(?:-image)?\s*:\s*url\(['\"]?([^'\"\)]+)['\"]?\)",
            tag.get("style", ""), _re.IGNORECASE,
        )
        for bg_url in bg_urls:
            bg_url = bg_url.strip()
            if bg_url and not bg_url.startswith("data:"):
                if not _AD_NETWORK_RE.search(bg_url):
                    _add(bg_url)

    return candidates


def _detect_logo_by_reference(
    soup, structured: dict, source: str, page_url: str, logo_ref_hashes: dict
) -> tuple:
    """
    Logo is PRESENT only if one of the candidate images on the page
    perceptually matches the user's uploaded reference logo.

    Returns (matched: bool, debug: dict) where debug includes the list of
    candidate URLs checked, whether each was downloadable, and the best
    Hamming distance found (lower = more similar; a match requires
    distance <= logo_matcher.DEFAULT_THRESHOLD).
    """
    import logo_matcher

    candidates = _collect_logo_candidate_urls(soup, structured, source)
    if not candidates:
        return False, {"candidates_found": 0, "checked": []}

    match, checked = logo_matcher.find_matching_image_debug(
        candidates, logo_ref_hashes, page_url=page_url
    )
    debug = {
        "candidates_found": len(candidates),
        "threshold": logo_matcher.DEFAULT_THRESHOLD,
        "matched_url": match,
        "checked": checked,
    }
    return match is not None, debug


def _detect_photos(soup, structured: dict, source: str = "") -> bool:
    brand = _directory_brand(source)

    def _is_own_brand_or_ad(url: str) -> bool:
        ul = url.lower()
        if _AD_NETWORK_RE.search(ul):
            return True
        if brand and brand in ul:
            return True
        return False

    if len(structured.get("image_urls", [])) >= 1:
        for url in structured["image_urls"]:
            if not _UI_IMAGE_PATTERNS.search(url) and not _is_own_brand_or_ad(url):
                return True

    hero_re = re.compile(
        r"(hero|banner|cover|carousel|slider|slideshow|gallery|"
        r"featured|backdrop|jumbotron|photo[-_]?section|media[-_]?section)",
        re.IGNORECASE,
    )
    for container in soup.find_all(["div", "section", "ul", "figure", "header"]):
        if not hero_re.search(_cls_str(container)):
            continue
        if _is_chrome_or_ad(container):
            continue
        for img in container.find_all("img"):
            srcs = _all_srcs(img)
            if srcs and not _is_tiny(img, 40) and not _UI_IMAGE_PATTERNS.search(srcs[0]) \
                    and not _is_own_brand_or_ad(srcs[0]):
                return True
        style = container.get("style", "")
        if "background" in style.lower() and "url(" in style.lower():
            return True

    for tag in soup.find_all(style=True):
        if _is_chrome_or_ad(tag):
            continue
        style_val = tag.get("style", "")
        bg_urls = re.findall(
            r"background(?:-image)?\s*:\s*url\(['\"]?([^'\"\)]+)['\"]?\)",
            style_val, re.IGNORECASE,
        )
        for url in bg_urls:
            if url.strip() and not _UI_IMAGE_PATTERNS.search(url) and not _is_own_brand_or_ad(url):
                return True

    for a in soup.find_all("a", href=True):
        if _is_chrome_or_ad(a):
            continue
        txt = a.get_text(strip=True).lower()
        href = a.get("href", "").lower()
        if txt in ("photos", "photo", "gallery", "images") or "photo" in href:
            if not any(d in href for d in ("google", "facebook", "twitter", "instagram")):
                return True

    large_count = 0
    for img in soup.find_all("img"):
        srcs = _all_srcs(img)
        if not srcs:
            continue
        src = srcs[0]
        if _UI_IMAGE_PATTERNS.search(src) or _is_tiny(img, 60):
            continue
        if any(x in src.lower() for x in ("favicon", "sprite", "icon-", "-icon")):
            continue
        if _is_chrome_or_ad(img) or _is_own_brand_or_ad(src):
            continue
        large_count += 1
        if large_count >= 2:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Keywords cleaner
# ─────────────────────────────────────────────────────────────────────────────

_ADDRESS_TOKEN_RE = re.compile(
    r"^\s*(\d{3,}.*|.*\b(blvd|boulevard|street|avenue|ave|road|lane|drive|"
    r"way|court|place|circle|parkway|highway|suite|ste)\b.*|"
    r"\d{5}(-\d{4})?|[a-z]{2}\s+\d{5})\s*$",
    re.IGNORECASE,
)
_KW_NOISE_RE = re.compile(
    r"^(address details|roadmap|satellite map|phone number|business hours|"
    r"trip planner?|travel|maps?|location|venue|place|trip)\s*$",
    re.IGNORECASE,
)
# US state names and abbreviations — used to filter location tags from keyword lists
_US_STATE_RE = re.compile(
    r"^(alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
    r"delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|"
    r"kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
    r"mississippi|missouri|montana|nebraska|nevada|new\s+hampshire|new\s+jersey|"
    r"new\s+mexico|new\s+york|north\s+carolina|north\s+dakota|ohio|oklahoma|"
    r"oregon|pennsylvania|rhode\s+island|south\s+carolina|south\s+dakota|"
    r"tennessee|texas|utah|vermont|virginia|washington|west\s+virginia|"
    r"wisconsin|wyoming|"
    r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|"
    r"MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|"
    r"VT|VA|WA|WV|WI|WY|DC)$",
    re.IGNORECASE,
)


def _clean_keywords(
    raw: str,
    business_name: str = "",
    category: str = "",
    city: str = "",
    state: str = "",
) -> str:
    """
    Clean a raw keyword/tag string.

    Splits on comma, pipe, or semicolon only (never spaces, to preserve
    multi-word tags). Drops tokens that are:
      - address-like / zip codes / bare state names or abbreviations
      - generic map/location noise
      - the business name itself (exact match)
      - the business's own city or state (exact match) — these are
        "Location tags", not keywords

    `category` is accepted for signature symmetry with callers but is not
    used to filter individual tokens here — see _universal_keywords()'s
    <meta name="keywords"> handling for the Category-duplicate guard.

    If, after filtering, nothing distinctive remains, returns "".
    """
    if not raw:
        return ""
    # Split on comma, pipe, or semicolon only — never on spaces
    tokens = [t.strip() for t in re.split(r"[,|;]", raw) if t.strip()]
    bn_lower    = business_name.lower().strip()
    cat_lower   = category.lower().strip()
    city_lower  = city.lower().strip()
    state_lower = state.lower().strip()
    cleaned = []
    for tok in tokens:
        tl = tok.lower().strip()
        if _ADDRESS_TOKEN_RE.match(tok):
            continue
        if _KW_NOISE_RE.match(tok):
            continue
        if re.search(r"\b\d{5}\b", tok):
            continue
        # Filter bare city, state combos (e.g. "Newark, DE" already split to "Newark" + "DE")
        if re.match(r"^[a-z\s]+,?\s+[a-z]{2}$", tok, re.IGNORECASE):
            continue
        if _US_STATE_RE.match(tok):
            continue
        if re.match(r"^[a-z]{2}$", tok, re.IGNORECASE):
            continue
        if re.match(r"^\d{5}(-\d{4})?$", tok):
            continue
        # Drop the business name itself
        if bn_lower and tl == bn_lower:
            continue
        # Drop the business's own city / state (these are location tags,
        # not keywords — e.g. brownbook's "Location tags" column)
        if city_lower and tl == city_lower:
            continue
        if state_lower and tl == state_lower:
            continue
        cleaned.append(tok)
    return ", ".join(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
#  Cloudflare / error page detector
# ─────────────────────────────────────────────────────────────────────────────

_CF_SIGNALS = (
    # Standard Cloudflare challenge pages
    "cloudflare.com?utm_source=challenge",
    "cf_chl_",
    "cdn-cgi/challenge-platform",
    "Just a moment",
    "checking your browser",
    "DDoS protection by Cloudflare",
    # Cloudflare 522 "Connection Timed Out" error pages
    "error 522",
    "522: connection timed out",
    "522 origin connection time-out",
    "contact your hosting provider",
    "your web server is not completing requests",
    "an error 522 means",
    "the request didn't finish",
    # Cloudflare 520 / 524 / 525 variants
    "error 520",
    "error 524",
    "error 525",
    "cloudflare ray id",
    # Generic origin unreachable
    "origin web server timed out",
    "the web server reported a bad gateway error",
)


def _is_cloudflare(html: str, text: str) -> bool:
    combined = (html[:5000] + text[:2000]).lower()
    return any(s.lower() in combined for s in _CF_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
#  URL slug parser (last-resort for JS-rendered pages with no content)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_url_slug(url: str) -> dict:
    out = {"name": "", "city": "", "state": "", "zip": ""}
    try:
        path = url.rstrip("/").split("?")[0]
        segments = [s for s in path.split("/") if s and not s.isdigit()]
        if not segments:
            return out
        slug = segments[-1]
        parts = slug.replace("_", "-").split("-")
        if len(parts) < 3:
            return out

        idx = len(parts) - 1
        if re.match(r"^\d{5}$", parts[idx]):
            out["zip"] = parts[idx]
            idx -= 1
        if idx >= 0 and re.match(r"^[A-Z]{2}$", parts[idx]):
            out["state"] = parts[idx]
            idx -= 1

        remaining = parts[:idx + 1]
        if remaining:
            out["city"] = remaining[-1]
            out["name"] = " ".join(remaining[:-1])
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Master pre-extraction (universal — no domain routing)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_hints(
    page_html: str,
    page_text: str,
    source: str = "",
    page_url: str = "",
    logo_ref_hashes: dict = None,
) -> dict:
    """
    Single universal extraction pass.
    Returns a hints dict consumed by build_prompt and post-processing.
    """
    hints = {
        "logo_confirmed":   False,
        "logo_html":        "",
        "photos_confirmed": False,
        "name":        "",
        "phone":       "",
        "email":       "",
        "street":      "",
        "city":        "",
        "state":       "",
        "zip":         "",
        "country":     "",
        "description": "",
        "website":     "",
        "hours":       "",
        "social":      "",
        "gbp":         "",
        "category":    "",
        "keywords":    "",
        "cloudflare_blocked": False,
    }

    if not page_html:
        return hints

    if _is_cloudflare(page_html, page_text):
        hints["cloudflare_blocked"] = True
        return hints

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page_html, "html.parser")

        # 1. Structured data (JSON-LD + microdata + meta) — highest confidence
        structured = _extract_structured(soup)

        # 2. Populate hints from structured data
        hints["name"]    = structured["name"]
        hints["phone"]   = structured["phone"]
        hints["email"]   = structured["email"]
        hints["street"]  = structured["street"]
        hints["city"]    = structured["city"]
        hints["state"]   = structured["state"]
        hints["zip"]     = structured["zip"]
        hints["country"] = structured["country"]

        # 3. Universal DOM heuristics to fill gaps
        hints["description"] = _universal_description(soup, structured["description"])
        hints["website"]     = structured["website"] or _universal_website(soup)
        hints["hours"]       = _universal_hours(soup, structured["hours"])
        hints["social"]      = _universal_social(soup)
        hints["gbp"]         = _universal_gbp(soup)
        hints["category"]    = structured["category"] or _universal_category(soup)
        hints["keywords"]    = _universal_keywords(
            soup,
            business_name=hints.get("name", ""),
            category=hints.get("category", ""),
            city=hints.get("city", ""),
            state=hints.get("state", ""),
        )

        # 4. Visual detection
        if logo_ref_hashes:
            logo_matched, _logo_debug = _detect_logo_by_reference(
                soup, structured, source, page_url, logo_ref_hashes
            )
            hints["logo_confirmed"] = logo_matched
        else:
            hints["logo_confirmed"] = _detect_logo(soup, structured, source)
        hints["photos_confirmed"] = _detect_photos(soup, structured, source)

    except Exception:
        pass

    return hints


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    page_text: str,
    page_html: str,
    fields: list,
    source: str = "",
    page_url: str = "",
    logo_ref_hashes: dict = None,
) -> str:
    hints = _extract_page_hints(page_html, page_text, source, page_url, logo_ref_hashes)

    if hints.get("cloudflare_blocked"):
        return (
            "You are a business data extraction assistant.\n"
            "This page returned a Cloudflare security challenge or connection error "
            "and contains NO real business data.\n"
            f"Return ONLY a JSON object with null for every field: {fields}\n"
            f"Example: {{{', '.join(repr(f)+': null' for f in fields)}}}"
        )

    facts = []

    if logo_ref_hashes:
        if hints["logo_confirmed"]:
            facts.append(
                'REFERENCE LOGO MATCH FOUND on this page — return "PRESENT" for Logo'
            )
        else:
            facts.append(
                'REFERENCE LOGO PROVIDED but NO MATCHING IMAGE found on this page — '
                'return null for Logo (this is authoritative; do not guess "PRESENT" '
                'based on other images)'
            )
    elif hints["logo_confirmed"]:
        facts.append('LOGO CONFIRMED PRESENT — return "PRESENT" for Logo')

    if hints["photos_confirmed"]:
        facts.append('PHOTOS CONFIRMED PRESENT — return "PRESENT" for Photos')

    field_hint_map = [
        ("name",        "NAME"),
        ("phone",       "PHONE"),
        ("email",       "BUSINESS EMAIL"),
        ("street",      "STREET"),
        ("city",        "CITY"),
        ("state",       "STATE"),
        ("zip",         "ZIPCODE"),
        ("country",     "COUNTRY"),
        ("description", "DESCRIPTION (use verbatim)"),
        ("website",     "WEBSITE URL"),
        ("hours",       "HOURS"),
        ("social",      "SOCIAL MEDIA LINKS"),
        ("gbp",         "GBP LINK"),
        ("category",    "CATEGORY"),
        ("keywords",    "KEYWORDS (use these verbatim)"),
    ]
    for hint_key, label in field_hint_map:
        val = hints.get(hint_key, "")
        if val:
            facts.append(f"{label}:\n{val}")

    if len(page_html) > 20000:
        html_snippet = page_html[:15000] + "\n\n…[middle omitted]…\n\n" + page_html[-5000:]
    else:
        html_snippet = page_html

    all_imgs = re.findall(r"<img[^>]*>", page_html, re.IGNORECASE)
    img_section = "\n".join(all_imgs[:100]) if all_imgs else ""

    bg_urls = re.findall(
        r"background(?:-image)?\s*:\s*url\(['\"]?([^'\"\)]+)['\"]?\)",
        page_html, re.IGNORECASE,
    )
    bg_section = "\n".join(bg_urls[:20]) if bg_urls else ""

    parts = []
    if facts:
        parts.append(
            "═══ PRE-EXTRACTED FIELDS (AUTHORITATIVE — use directly, do NOT override) ═══\n\n"
            + "\n\n".join(facts)
        )
    if page_text and len(page_text.strip()) > 100:
        parts.append(f"PAGE TEXT:\n{page_text[:30000]}")
    parts.append(f"PAGE HTML SNIPPET:\n{html_snippet}")
    if img_section:
        parts.append(f"ALL <img> TAGS:\n{img_section}")
    if bg_section:
        parts.append(f"CSS BACKGROUND-IMAGE URLs:\n{bg_section}")

    content = "\n\n".join(parts)

    field_rules = []
    for f in fields:
        if f == "Name":
            field_rules.append('- "Name": primary business name only. Never return a website domain or URL.')
        elif f == "Phone":
            field_rules.append('- "Phone": phone number — digits and separators only.')
        elif f == "Website URL":
            field_rules.append(
                '- "Website URL": the business\'s own website.\n'
                '  Use PRE-EXTRACTED if present. Find "Visit Website" or external link.\n'
                '  NEVER return a cloudflare.com or directory domain URL.'
            )
        elif f == "Street":
            field_rules.append('- "Street": street address (number + street name).')
        elif f == "City":
            field_rules.append('- "City": city name.')
        elif f == "State":
            field_rules.append('- "State": state/region as shown.')
        elif f == "Zipcode":
            field_rules.append('- "Zipcode": postal/zip code.')
        elif f == "Country":
            field_rules.append('- "Country": country name or ISO code.')
        elif f == "Category":
            field_rules.append('- "Category": business type or industry. Use PRE-EXTRACTED if present.')
        elif f == "Keywords":
            field_rules.append(
                '- "Keywords": a dedicated tags/keywords list — NOT the same thing as Category.\n'
                '  Use KEYWORDS from PRE-EXTRACTED FIELDS verbatim if present.\n'
                '  Otherwise look ONLY for an explicit "Tags", "Keywords", or "Business tags"\n'
                '  section, or pipe/comma-separated label list (e.g. "AI | Legal | Law").\n'
                '  Do NOT invent Keywords from the Category, business name, page title,\n'
                '  or address/city/state. If the page has no distinct tags/keywords section\n'
                '  separate from Category, return null — do NOT repeat the Category value\n'
                '  (or the business name, or the city/state) as Keywords.'
            )
        elif f == "Description":
            field_rules.append(
                '- "Description": the business\'s own descriptive prose.\n'
                '  Use DESCRIPTION from PRE-EXTRACTED FIELDS verbatim if present.\n'
                '  Otherwise find multi-sentence paragraphs describing what the business does —\n'
                '  look especially in "More about …" sections at the bottom of the page.\n'
                '  NEVER return: addresses, phone numbers, navigation breadcrumbs,\n'
                '  category labels (e.g. "BusinessName Category: Fire & Water Damage Repair"),\n'
                '  page titles, or single-line classification strings.\n'
                '  A valid description is at least 2 sentences of genuine business prose.\n'
                '  Return null only if no such prose exists on the page.'
            )
        elif f == "Hours":
            field_rules.append(
                '- "Hours": operating hours.\n'
                '  Use HOURS from PRE-EXTRACTED FIELDS if present.\n'
                '  Look for a table or list showing day names and times (e.g. "Monday 9am-5pm").\n'
                '  NEVER return hours where every time value is "00:00" — those are\n'
                '  unrendered JS placeholders; return null instead.\n'
                '  NEVER return an address, phone number, or review text as hours.'
            )
        elif f == "Social Media Links":
            field_rules.append(
                '- "Social Media Links": Facebook, LinkedIn, Twitter/X, Instagram, YouTube, etc.\n'
                '  Use PRE-EXTRACTED if present. Return comma-separated URLs.'
            )
        elif f == "GBP Link":
            field_rules.append('- "GBP Link": Google Business Profile / Maps link.')
        elif f == "Business Email":
            field_rules.append('- "Business Email": email address on the page.')
        elif f == "Logo":
            field_rules.append(
                '- "Logo": does a business logo exist?\n'
                '  If LOGO CONFIRMED PRESENT in PRE-EXTRACTED FIELDS → return "PRESENT".\n'
                '  Otherwise check <img> tags and CSS backgrounds.\n'
                '  Return "PRESENT" or null.'
            )
        elif f == "Photos":
            field_rules.append(
                '- "Photos": do business photos exist?\n'
                '  If PHOTOS CONFIRMED PRESENT in PRE-EXTRACTED FIELDS → return "PRESENT".\n'
                '  Otherwise check hero/banner/gallery sections.\n'
                '  Return "PRESENT" or null.'
            )

    rules_text = "\n".join(field_rules)

    return f"""You are a business data extraction assistant. Extract listing information from the content below.

Extract ONLY these fields: {fields}

CRITICAL RULES:
- Return ONLY a valid JSON object. No markdown, no backticks, no explanation.
- PRE-EXTRACTED FIELDS are AUTHORITATIVE: use those values directly without modification.
- Use null for fields genuinely absent from the page.
- Do NOT guess or invent values.
- Logo/Photos: if PRE-EXTRACTED confirms PRESENT, return "PRESENT" — do not second-guess.
- Website URL: NEVER return a cloudflare.com URL.
- Hours: NEVER return "00:00 to 00:00" placeholders — return null instead.
- Keywords: use KEYWORDS from PRE-EXTRACTED FIELDS verbatim if present. Otherwise search the page for an explicit "Tags"/"Keywords"/"Business tags" section or pipe/comma-separated label list and return comma-separated. NEVER return the Category value, business name, or city/state as Keywords — if no distinct tags section exists, return null.
- Name: NEVER return a website domain (e.g. "nearfinderus.com") as the business name.

FIELD INSTRUCTIONS:
{rules_text}

{content}"""


# ─────────────────────────────────────────────────────────────────────────────
#  Gemini API
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
    return all(extracted.get(f) in (None, "", "null") for f in fields)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_fields(
    page_text: str,
    page_html: str,
    fields: list,
    source: str,
    api_key: str,
    page_url: str = "",
    logo_ref_hashes: dict = None,
) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(page_text, page_html, fields, source, page_url, logo_ref_hashes)

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

        # ── Post-process: enforce authoritative hints ──────────────────────
        hints = _extract_page_hints(page_html, page_text, source, page_url, logo_ref_hashes)

        if hints.get("cloudflare_blocked"):
            for f in fields:
                extracted[f] = None
            extracted["_cloudflare_blocked"] = True
            extracted["_model"] = model_used
            return extracted

        def _empty(v):
            return v in (None, "", "null", "None")

        # Visual fields
        if "Logo" in fields:
            if logo_ref_hashes:
                # Reference logo provided — exact-match result is authoritative
                # in BOTH directions (overrides whatever Gemini guessed).
                extracted["Logo"] = "PRESENT" if hints["logo_confirmed"] else None
            elif hints["logo_confirmed"]:
                extracted["Logo"] = "PRESENT"
        if hints["photos_confirmed"] and "Photos" in fields:
            extracted["Photos"] = "PRESENT"

        # All text/URL fields — enforce hint when AI returned empty
        hint_field_map = [
            ("description", "Description"),
            ("website",     "Website URL"),
            ("hours",       "Hours"),
            ("social",      "Social Media Links"),
            ("gbp",         "GBP Link"),
            ("category",    "Category"),
            ("name",        "Name"),
            ("phone",       "Phone"),
            ("email",       "Business Email"),
            ("street",      "Street"),
            ("city",        "City"),
            ("state",       "State"),
            ("zip",         "Zipcode"),
            ("country",     "Country"),
        ]
        for hint_key, field_name in hint_field_map:
            if hints.get(hint_key) and field_name in fields:
                if _empty(extracted.get(field_name)):
                    extracted[field_name] = hints[hint_key]

        # Keywords: use pre-extracted hint as fallback if Gemini returned empty
        # (hints["keywords"] is already cleaned via _universal_keywords, which
        # was called with business_name/category/city/state context)
        if "Keywords" in fields:
            kw_hint = hints.get("keywords", "")
            if _empty(extracted.get("Keywords")) and kw_hint:
                extracted["Keywords"] = kw_hint

        # ── Final guards ───────────────────────────────────────────────────

        # Name: reject if it looks like a domain name (e.g. "nearfinderus.com")
        if "Name" in fields:
            name_val = extracted.get("Name", "") or ""
            if name_val and _DOMAIN_NAME_RE.search(name_val):
                hint_name = hints.get("name", "")
                # Use hint only if it doesn't also look like a domain
                extracted["Name"] = (
                    hint_name
                    if hint_name and not _DOMAIN_NAME_RE.search(hint_name)
                    else None
                )

        # Website: strip cloudflare URLs
        if "Website URL" in fields:
            url_val = extracted.get("Website URL", "") or ""
            if "cloudflare.com" in url_val:
                extracted["Website URL"] = hints.get("website") or None

        # Hours: reject placeholder / phone / review text
        if "Hours" in fields:
            hours_val = extracted.get("Hours", "") or ""
            if hours_val:
                _ph_re = re.compile(r"0{1,2}:0{2}\s*(?:to|-|–)\s*0{1,2}:0{2}", re.IGNORECASE)
                real_times = re.findall(r"\d{1,2}:\d{2}", hours_val)
                placeholder_times = _ph_re.findall(hours_val)
                if _PHONE_IN_TEXT.search(hours_val):
                    extracted["Hours"] = None
                elif re.search(r"\breview\b", hours_val, re.IGNORECASE) and not re.search(r"\d{1,2}:\d{2}", hours_val):
                    extracted["Hours"] = None
                elif real_times and all(re.match(r"^0{1,2}:0{2}$", t) for t in real_times):
                    extracted["Hours"] = None
                    if hints.get("hours"):
                        extracted["Hours"] = hints["hours"]
                elif len(placeholder_times) > 0 and len(placeholder_times) >= len(real_times):
                    extracted["Hours"] = None

        # Description: reject false descriptions (breadcrumbs, category strings, etc.)
        if "Description" in fields:
            desc_val = extracted.get("Description", "") or ""
            if desc_val:
                if not _good_desc(desc_val) or _is_false_description(desc_val):
                    hint_desc = hints.get("description", "")
                    extracted["Description"] = (
                        hint_desc if (_good_desc(hint_desc) and not _is_false_description(hint_desc))
                        else None
                    )

        # Keywords: clean whatever remains — strip business name, category,
        # and city/state from any meta-keywords-style blob (the structural
        # "Business tags" / "Tags" extraction in _universal_keywords already
        # excludes Category-only single-token results — see Priority 1 there).
        if "Keywords" in fields:
            kw_val = extracted.get("Keywords", "") or ""
            if _empty(kw_val):
                extracted["Keywords"] = None
            else:
                bn  = extracted.get("Name", "") or ""
                cat = extracted.get("Category", "") or ""
                cty = extracted.get("City", "") or ""
                st  = extracted.get("State", "") or ""
                cleaned_kw = _clean_keywords(
                    kw_val, business_name=bn, category=cat, city=cty, state=st
                )
                extracted["Keywords"] = cleaned_kw or None

        # All-null fallback: try URL slug
        if _all_null(extracted, fields) and source:
            slug = _parse_url_slug(source)
            for field_name, slug_key in [("Name", "name"), ("City", "city"),
                                          ("State", "state"), ("Zipcode", "zip")]:
                if slug.get(slug_key) and field_name in fields:
                    extracted[field_name] = slug[slug_key]
            if any(slug.get(k) for k in ("name", "city", "state", "zip")):
                extracted["_slug_fallback"] = True

        if _all_null(extracted, fields) and (page_html or page_text):
            extracted["_all_null_warning"] = (
                "Gemini returned null for all fields despite page content being present."
            )

    except json.JSONDecodeError as e:
        extracted = {"_parse_error": str(e), "_raw": raw[:800], "_model": model_used}
        try:
            hints = _extract_page_hints(page_html, page_text, source, page_url, logo_ref_hashes)
            if "Logo" in fields:
                if logo_ref_hashes:
                    extracted["Logo"] = "PRESENT" if hints["logo_confirmed"] else None
                elif hints["logo_confirmed"]:
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
    logo_ref_hashes: dict = None,
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
                cleaned_text, cleaned_html, fields, source, api_key,
                page_url=page.get("url", ""),
                logo_ref_hashes=logo_ref_hashes,
            )

        result["_url"] = page["url"]
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, len(scraped_pages))

    return results

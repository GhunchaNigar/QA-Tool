"""
comparator.py
Compares user-provided business data against AI-extracted page data.
Returns CORRECT / INCORRECT / MISSING / N/A for each field.

Each URL is compared using only the fields defined for its detected source.
"""

import re
import unicodedata
from fields_config import NA_OVERRIDES, VISUAL_FIELDS, ALL_FIELDS


# ── US State lookup ───────────────────────────────────────────────────────────

_US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Reverse map: full name -> abbreviation
_US_STATE_ABBREVS = {v: k for k, v in _US_STATES.items()}


# ── Common English stop words ─────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "has",
    "have", "had", "do", "does", "did", "that", "this", "it", "its",
    "we", "our", "you", "your", "they", "their", "as", "from", "up",
    "not", "no", "can", "will", "about", "into", "than", "then", "so",
}

# Legal suffixes stripped from business names
_LEGAL_SUFFIXES = re.compile(
    r"\b(llc|inc|ltd|co|corp|corporation|company|group|plc|lp|llp|pllc)\b\.?",
    re.IGNORECASE,
)


# ── Normalizers ───────────────────────────────────────────────────────────────

def _to_ascii(value: str) -> str:
    """Normalize unicode to closest ASCII equivalent."""
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def normalize_phone(value: str) -> str:
    """Strip everything except digits."""
    return re.sub(r"\D", "", value)


def normalize_url(value: str) -> str:
    """Strip protocol, www prefix, trailing slash, and lowercase."""
    v = value.lower().strip()
    v = re.sub(r"^https?://", "", v)
    v = re.sub(r"^www\.", "", v)
    v = v.rstrip("/")
    return v


def normalize_name(value: str) -> str:
    """
    Lowercase, ASCII-fold, strip legal suffixes, punctuation, and extra spaces.
    Does NOT do substring matching — callers use tokens_match() for names.
    """
    v = _to_ascii(value).lower().strip()
    v = _LEGAL_SUFFIXES.sub("", v)
    v = re.sub(r"[&'\"`,\.;\-–—]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def normalize_state(value: str) -> str:
    """
    Always resolve to the full lowercase state name so that abbreviations
    and full names compare equal.
      'NJ'          -> 'new jersey'
      'New Jersey'  -> 'new jersey'
      'new jersey'  -> 'new jersey'
    Falls back to the cleaned raw string for unrecognised values.
    """
    v = value.strip().lower()
    # Abbreviation -> full name
    if v in _US_STATES:
        return _US_STATES[v]
    # Already a full name
    if v in _US_STATE_ABBREVS:
        return v
    # Unknown — return as-is (cleaned)
    return v


def normalize_address(value: str) -> str:
    """Lowercase, expand common abbreviations, strip punctuation."""
    v = _to_ascii(value).lower().strip()
    abbrevs = {
        r"\bst\b":    "street",
        r"\bave\b":   "avenue",
        r"\bblvd\b":  "boulevard",
        r"\brd\b":    "road",
        r"\bdr\b":    "drive",
        r"\bln\b":    "lane",
        r"\bct\b":    "court",
        r"\bpl\b":    "place",
        r"\bsuite\b": "ste",
    }
    for pattern, replacement in abbrevs.items():
        v = re.sub(pattern, replacement, v)
    v = re.sub(r"[,\.;\-#]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def normalize_country(value: str) -> str:
    """
    Resolve any country name, alpha-2, or alpha-3 code to a lowercase alpha-2
    using the pycountry library — no hardcoded country lists.
    Falls back to the cleaned raw string if unrecognised.
    """
    import pycountry
    _NON_STANDARD = {"uk": "gb"}  # only genuinely non-ISO common codes
    v = value.strip()
    if not v:
        return ""
    for lookup in [
        lambda: pycountry.countries.get(alpha_2=v.upper()),
        lambda: pycountry.countries.get(alpha_3=v.upper()),
        lambda: pycountry.countries.lookup(v),
    ]:
        try:
            result = lookup()
            if result:
                return result.alpha_2.lower()
        except LookupError:
            pass
    return _NON_STANDARD.get(v.lower(), v.lower().strip())


def normalize_email(value: str) -> str:
    return value.lower().strip()


def normalize_general(value: str) -> str:
    v = _to_ascii(value).lower().strip()
    v = re.sub(r"[,\.;:\-\/\\|]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def normalize(value: str, field: str = "") -> str:
    if not value:
        return ""
    fl = field.lower()
    if "phone" in fl:
        return normalize_phone(value)
    if "url" in fl or "website" in fl:
        return normalize_url(value)
    if fl == "name":
        return normalize_name(value)
    if fl == "country":
        return normalize_country(value)
    if fl == "state":
        return normalize_state(value)
    if fl in ("street", "city", "zipcode"):
        return normalize_address(value)
    if "email" in fl:
        return normalize_email(value)
    return normalize_general(value)


# ── Token helpers ─────────────────────────────────────────────────────────────

def _meaningful_tokens(text: str) -> set:
    """Split on whitespace and drop stop words and very short tokens."""
    return {w for w in text.split() if len(w) > 2 and w not in _STOP_WORDS}


def _token_overlap(a: str, b: str) -> float:
    """
    Jaccard-style overlap: intersection / len(user_tokens).
    Returns 0.0 if user has no meaningful tokens.
    """
    ta = _meaningful_tokens(a)
    tb = _meaningful_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


# ── Field-specific matchers ───────────────────────────────────────────────────

def _match_phone(u: str, e: str) -> bool:
    """Match on last 9 digits (handles country-code differences)."""
    if not u or not e:
        return False
    if u == e:
        return True
    if len(u) >= 9 and len(e) >= 9:
        return u[-9:] == e[-9:]
    return False


def _match_url(u: str, e: str) -> bool:
    """Exact match after normalization. Substring only if one is a path prefix of the other."""
    if not u or not e:
        return False
    if u == e:
        return True
    u_domain = u.split("/")[0]
    e_domain = e.split("/")[0]
    return u_domain == e_domain


def _match_name(u: str, e: str) -> bool:
    """
    Names match if:
      - Exact after normalization, OR
      - Every token in the shorter name appears in the longer name
        (catches "Focal" vs "Focal Agency" only when user typed the full name).
    We do NOT use simple `u in e` to avoid "Al" matching "Alabama".
    """
    if not u or not e:
        return False
    if u == e:
        return True
    tu = set(u.split())
    te = set(e.split())
    if not tu or not te:
        return False
    shorter, longer = (tu, te) if len(tu) <= len(te) else (te, tu)
    if shorter <= longer:
        if len(shorter) >= 2:
            return True
        token = next(iter(shorter))
        return len(token) >= 5
    return False


def _match_address_field(u: str, e: str) -> bool:
    """Exact match after normalization. Substring OK for street (unit numbers)."""
    if not u or not e:
        return False
    return u == e or u in e or e in u


def _match_keywords(u: str, e: str) -> bool:
    """
    Keywords are comma/pipe-separated lists.
    Match if ≥60% of user's keywords appear in extracted text.
    """
    def parse_kw(s):
        return {k.strip().lower() for k in re.split(r"[,|;]", s) if k.strip()}

    uk = parse_kw(u)
    ek_text = e.lower()
    if not uk:
        return False
    matched = sum(1 for kw in uk if kw in ek_text)
    return (matched / len(uk)) >= 0.60


def _match_description(u: str, e: str) -> bool:
    """
    Descriptions match if:
    - ≥55% of meaningful user words appear in extracted text, OR
    - The extracted text is a leading substring of the user description
      (handles "See More" truncation — site only shows first N chars).
    """
    overlap = _token_overlap(u, e)
    if overlap >= 0.55:
        return True
    if len(e) > 50 and u.startswith(e[:min(len(e), 80)]):
        return True
    e_tokens = _meaningful_tokens(e)
    u_tokens = _meaningful_tokens(u)
    if e_tokens and u_tokens:
        reverse_overlap = len(e_tokens & u_tokens) / len(e_tokens)
        if reverse_overlap >= 0.80 and len(e_tokens) >= 10:
            return True
    return False


def _match_hours(u: str, e: str) -> bool:
    """
    Hours: extract time-like tokens (digits, am/pm, day names) and compare.
    Match if ≥50% of user time tokens found in extracted hours.
    """
    _time_token = re.compile(
        r"\b(\d{1,2}(?::\d{2})?(?:am|pm)?|mon|tue|wed|thu|fri|sat|sun|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"open|closed|24)\b",
        re.IGNORECASE,
    )
    ut = set(t.lower() for t in _time_token.findall(u))
    et = set(t.lower() for t in _time_token.findall(e))
    if not ut:
        return _token_overlap(u, e) >= 0.40
    if not et:
        return False
    return len(ut & et) / len(ut) >= 0.50


def _match_social(u: str, e: str) -> bool:
    """
    Social links: extract domain names from URLs and check overlap.
    """
    def extract_domains(text):
        return set(re.findall(
            r"(facebook|twitter|linkedin|instagram|youtube|tiktok|pinterest)",
            text.lower()
        ))
    ud = extract_domains(u)
    ed = extract_domains(e)
    if not ud:
        return _token_overlap(u, e) >= 0.50
    return bool(ud & ed)


def _match_category(u: str, e: str) -> bool:
    """Categories: token overlap ≥50%, ignoring stop words."""
    if u == e:
        return True
    return _token_overlap(u, e) >= 0.50


_CATEGORY_SPLIT_RE = re.compile(r"[|;\n]+")


def _match_category_multi(user_val: str, extracted_val: str) -> bool:
    """
    The user may provide up to 4 categories (joined by " | ").
    CORRECT if ANY one of the user's categories matches the extracted value.
    INCORRECT only if none match.
    """
    e_norm = normalize_general(str(extracted_val))
    if not e_norm:
        return False

    candidates = [c.strip() for c in _CATEGORY_SPLIT_RE.split(str(user_val)) if c.strip()]
    if not candidates:
        return False

    for cat in candidates:
        u_norm = normalize_general(cat)
        if u_norm and _match_category(u_norm, e_norm):
            return True
    return False


# ── Main matching dispatcher ──────────────────────────────────────────────────

def values_match(user_val: str, extracted_val: str, field: str) -> bool:
    """
    Route to the right matcher based on field type.
    """
    fl = field.lower()

    # Category handled before normalization — user value may contain multiple
    # "|"-separated entries; any single match counts as CORRECT.
    if "category" in fl:
        return _match_category_multi(user_val, extracted_val)

    u = normalize(str(user_val), field)
    e = normalize(str(extracted_val), field)

    if not u or not e:
        return False

    if "phone" in fl:
        return _match_phone(u, e)

    if "url" in fl or "website" in fl:
        return _match_url(u, e)

    if fl == "name":
        return _match_name(u, e)

    if fl in ("street", "city", "zipcode", "country"):
        return _match_address_field(u, e)

    # State: both sides are already normalized to full name by normalize_state(),
    # so a simple equality check is sufficient.
    if fl == "state":
        return u == e

    if "email" in fl:
        return u == e  # emails must be exact

    if "keyword" in fl:
        return _match_keywords(user_val, extracted_val)  # raw values

    if "description" in fl or "about" in fl:
        return _match_description(u, e)

    if "hour" in fl:
        return _match_hours(user_val, extracted_val)  # raw values preserve time tokens

    if "social" in fl:
        return _match_social(user_val, extracted_val)

    # Fallback: exact normalized match
    return u == e


# ── Row comparison ────────────────────────────────────────────────────────────

def compare_row(
    user_data: dict, extracted_data: dict, source_fields: list, source: str
) -> dict:
    """
    Compare a single URL's extracted data against user-provided values.
    source_fields : fields applicable to this URL's detected source.
    Returns dict with ALL_FIELDS -> status string and overall "Status".
    """
    na_fields    = set(NA_OVERRIDES.get(source, []))
    row_result   = {}
    has_error    = False
    scrape_error = extracted_data.get("_scrape_error")

    for field in ALL_FIELDS:

        # Field not tracked for this source → N/A
        if field not in source_fields:
            row_result[field] = "N/A"
            continue

        # Hardcoded N/A override for this source
        if field in na_fields:
            row_result[field] = "N/A"
            continue

        # Whole page failed to scrape
        if scrape_error:
            row_result[field] = "SCRAPE ERROR"
            has_error = True
            continue

        user_val      = user_data.get(field, "").strip()
        extracted_val = extracted_data.get(field)

        # User left this field blank → skip
        if not user_val:
            row_result[field] = "N/A"
            continue

        # Visual fields: Logo / Photos → presence check only
        if field in VISUAL_FIELDS:
            if extracted_val == "PRESENT":
                row_result[field] = "CORRECT"
            else:
                row_result[field] = "MISSING"
                has_error = True
            continue

        # AI returned null / empty
        if extracted_val is None or str(extracted_val).strip() in ("", "null", "None"):
            row_result[field] = "MISSING"
            has_error = True
            continue

        # Compare via field-aware matcher
        if values_match(user_val, str(extracted_val), field):
            row_result[field] = "CORRECT"
        else:
            row_result[field] = "INCORRECT"
            has_error = True

    row_result["Status"] = "INCORRECT" if has_error else "CORRECT"
    return row_result


def compare_all(
    user_data: dict,
    extracted_list: list,
    url_source_map: dict,
    source_fields_map: dict,
) -> list:
    """
    Compare all extracted pages against user data.
    url_source_map    : {url -> detected_source_key}
    source_fields_map : {source_key -> [fields]}
    Returns list of result dicts (one per URL).
    """
    results = []
    for extracted in extracted_list:
        url    = extracted.get("_url", "")
        source = url_source_map.get(url, "unknown")
        fields = source_fields_map.get(source, [])

        comparison = compare_row(user_data, extracted, fields, source)
        comparison["Live Link"] = url
        comparison["Source"]    = source
        results.append(comparison)
    return results

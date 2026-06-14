import io
from urllib.parse import urljoin

import requests
from PIL import Image
import imagehash


# Hash functions to compute / compare. Using several reduces false
# negatives from any single algorithm's blind spots.
_HASH_FUNCS = {
    "phash":   imagehash.phash,
    "ahash":   imagehash.average_hash,
    "dhash":   imagehash.dhash,
}

# Hamming distance (out of 64 bits). 0 = pixel-perceptually identical.
# A small threshold still allows for resizing / re-compression artifacts
# while remaining a "this is the same image" match rather than "similar".
DEFAULT_THRESHOLD = 5

# Don't bother downloading anything absurdly large (logos are small).
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def compute_reference_hashes(image_bytes: bytes) -> dict:
    """
    Compute perceptual hashes for the user's reference logo.
    Returns {} if the bytes aren't a valid image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {}

    hashes = {}
    for name, fn in _HASH_FUNCS.items():
        try:
            hashes[name] = fn(img)
        except Exception:
            continue
    return hashes


def _fetch_image_bytes(url: str, timeout: int = 8) -> bytes | None:
    if not url or not url.startswith("http"):
        return None
    try:
        resp = requests.get(
            url, timeout=timeout, headers=_HEADERS, stream=True
        )
        if resp.status_code != 200:
            return None
        content = b""
        for chunk in resp.iter_content(chunk_size=65536):
            content += chunk
            if len(content) > MAX_IMAGE_BYTES:
                return None
        return content or None
    except Exception:
        return None


def image_bytes_match(
    image_bytes: bytes, ref_hashes: dict, threshold: int = DEFAULT_THRESHOLD
) -> bool:
    """True if `image_bytes` perceptually matches the reference logo."""
    if not ref_hashes:
        return False
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False

    for name, fn in _HASH_FUNCS.items():
        ref = ref_hashes.get(name)
        if ref is None:
            continue
        try:
            candidate_hash = fn(img)
        except Exception:
            continue
        if (candidate_hash - ref) <= threshold:
            return True
    return False


def url_matches_reference(
    image_url: str,
    ref_hashes: dict,
    page_url: str = "",
    threshold: int = DEFAULT_THRESHOLD,
) -> bool:
    """
    Download `image_url` (resolved against `page_url` if relative) and
    check whether it perceptually matches the reference logo.
    """
    if not ref_hashes or not image_url:
        return False
    if image_url.startswith("//"):
        image_url = "https:" + image_url
    elif not image_url.startswith("http") and page_url:
        image_url = urljoin(page_url, image_url)
    if not image_url.startswith("http"):
        return False

    data = _fetch_image_bytes(image_url)
    if not data:
        return False
    return image_bytes_match(data, ref_hashes, threshold)


def find_matching_image(
    candidate_urls: list,
    ref_hashes: dict,
    page_url: str = "",
    threshold: int = DEFAULT_THRESHOLD,
) -> str | None:
    """
    Check ALL candidate image URLs against the reference logo.
    Returns the first matching URL, or None if no match is found.
    Every candidate is checked — no early cutoff.
    """
    if not ref_hashes:
        return None
    for url in candidate_urls:
        if url_matches_reference(url, ref_hashes, page_url, threshold):
            return url
    return None


def find_matching_image_debug(
    candidate_urls: list,
    ref_hashes: dict,
    page_url: str = "",
    threshold: int = DEFAULT_THRESHOLD,
) -> tuple:
    """
    Check ALL candidate image URLs against the reference logo.
    Returns (matched_url | None, checked_list) where checked_list is a list
    of dicts with keys: url, downloaded, best_distance, matched.

    Every candidate is checked — no early cutoff — so no image is skipped.
    Once a match is found the URL is returned immediately but the loop
    still records the result for the debug output.
    """
    if not ref_hashes:
        return None, []

    checked = []
    matched_url = None

    for url in candidate_urls:
        entry = {"url": url, "downloaded": False, "best_distance": None, "matched": False}

        # Resolve relative / protocol-relative URLs
        resolved = url
        if resolved.startswith("//"):
            resolved = "https:" + resolved
        elif not resolved.startswith("http") and page_url:
            from urllib.parse import urljoin
            resolved = urljoin(page_url, resolved)

        if not resolved.startswith("http"):
            checked.append(entry)
            continue

        data = _fetch_image_bytes(resolved)
        if not data:
            checked.append(entry)
            continue

        entry["downloaded"] = True

        # Compute best Hamming distance across all hash functions
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            checked.append(entry)
            continue

        best_dist = None
        is_match = False
        for name, fn in _HASH_FUNCS.items():
            ref = ref_hashes.get(name)
            if ref is None:
                continue
            try:
                dist = fn(img) - ref
            except Exception:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
            if dist <= threshold:
                is_match = True

        entry["best_distance"] = best_dist
        entry["matched"] = is_match
        checked.append(entry)

        if is_match and matched_url is None:
            matched_url = resolved  # record first match; continue checking rest

    return matched_url, checked

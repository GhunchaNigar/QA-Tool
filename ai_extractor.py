"""
ai_extractor.py
Sends scraped page content to Google Gemini and extracts
business fields as JSON — including visual fields (Logo, Photos).
No hardcoded layout assumptions. Gemini searches the whole page.
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


def _extract_nearfinder_hints(page_html: str, page_text: str) -> dict:
    """
    Structurally extract logo, description, and website URL from nearfinderus HTML.
    Uses the actual DOM patterns found on nearfinderus.com listing pages —
    no keyword guessing, no truncation risk.

    Patterns confirmed from live HTML:
    - Logo: <figure class="lazy-element lazy-element--rendered"><img ...>
    - Description: <div class="mt-4"> ... <p>text</p> (first visible p, not inside display:none)
    - Website URL: <a href="/en/empresa/redirect?url=ENCODED_URL&...">
    - Social/WhatsApp: <a href="/en/empresa/redirect?url=https://api.whatsapp.com/...">
    - GBP Link: <a href="https://www.google.com.br/maps/place/...">
    """
    hints = {
        "logo_html": "",
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

        # ── Logo ──────────────────────────────────────────────────────────
        # nearfinderus renders the business logo inside a lazy-element--rendered figure
        for fig in soup.find_all("figure", class_=lambda c: c and "lazy-element--rendered" in " ".join(c)):
            img = fig.find("img")
            if img and img.get("src", ""):
                hints["logo_html"] = str(img)
                break
        # Fallback: any img with /logos/ or /thumb_ in src (nearfinderus logo URL pattern)
        if not hints["logo_html"]:
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if "/logos/" in src or "/thumb_" in src:
                    hints["logo_html"] = str(img)
                    break

        # ── Description ───────────────────────────────────────────────────
        # nearfinderus puts description in <div class="mt-4"> → <p> (visible only)
        desc_container = soup.find("div", class_="mt-4")
        if desc_container:
            for p in desc_container.find_all("p"):
                # Skip paragraphs inside display:none parents (the hidden "See More" block)
                hidden = False
                for parent in p.parents:
                    style = parent.get("style", "") if hasattr(parent, "get") else ""
                    if "display: none" in style or "display:none" in style:
                        hidden = True
                        break
                if not hidden:
                    text = p.get_text(separator=" ", strip=True)
                    if len(text) > 30:
                        hints["description_text"] = text
                        break

        # ── Website URL ───────────────────────────────────────────────────
        # nearfinderus wraps outbound links in /en/empresa/redirect?url=ENCODED
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/empresa/redirect?url=" in href:
                raw = href.split("url=")[1].split("&")[0]
                decoded = unquote(raw)
                # Exclude WhatsApp and internal/social links
                if "whatsapp.com" not in decoded and "nearfinderus.com" not in decoded:
                    hints["website_url"] = decoded
                    break

        # ── Social Media Links ────────────────────────────────────────────
        social_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/empresa/redirect?url=" in href:
                raw = href.split("url=")[1].split("&")[0]
                decoded = unquote(raw)
                if any(s in decoded for s in ["whatsapp.com", "facebook.com", "instagram.com",
                                               "linkedin.com", "twitter.com", "youtube.com", "tiktok.com"]):
                    social_links.append(decoded)
        if social_links:
            hints["social_links"] = ", ".join(social_links)

        # ── GBP Link ──────────────────────────────────────────────────────
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "google.com" in href and "maps/place" in href:
                hints["gbp_link"] = href
                break

    except Exception:
        pass

    return hints


def build_prompt(page_text: str, page_html: str, fields: list) -> str:
    """
    Build a generic extraction prompt.
    Gemini searches the entire page content for each requested field.
    HTML is always included so the model can detect images for visual fields.
    """
    import re as _re

    # Pre-extract structured values from HTML using BeautifulSoup
    hints = _extract_nearfinder_hints(page_html, page_text)

    # Extract ALL img tags from full HTML — for logo/photo detection
    all_img_tags = _re.findall(r'<img[^>]*>', page_html, flags=_re.IGNORECASE)
    img_section = "\n".join(all_img_tags[:80]) if all_img_tags else ""

    # HTML snippet: first 15k + last 5k
    if len(page_html) > 20000:
        html_snippet = page_html[:15000] + "\n\n…[middle omitted]…\n\n" + page_html[-5000:]
    else:
        html_snippet = page_html

    # Build pre-extracted facts section — these are DEFINITIVE values, not hints
    pre_extracted_facts = []
    if hints["logo_html"]:
        pre_extracted_facts.append(f'LOGO IMAGE FOUND IN HTML (return "PRESENT" for Logo field):\n{hints["logo_html"]}')
    if hints["description_text"]:
        pre_extracted_facts.append(f'DESCRIPTION TEXT (use this for Description field):\n{hints["description_text"]}')
    if hints["website_url"]:
        pre_extracted_facts.append(f'WEBSITE URL (use this for Website URL field):\n{hints["website_url"]}')
    if hints["social_links"]:
        pre_extracted_facts.append(f'SOCIAL MEDIA LINKS (use this for Social Media Links field):\n{hints["social_links"]}')
    if hints["gbp_link"]:
        pre_extracted_facts.append(f'GBP LINK (use this for GBP Link field):\n{hints["gbp_link"]}')

    # Build content section
    parts = []
    if pre_extracted_facts:
        parts.append(
            "PRE-EXTRACTED FIELDS (these are confirmed values directly from the page HTML — "
            "use them as-is for the corresponding fields):\n\n"
            + "\n\n".join(pre_extracted_facts)
        )
    if page_text and len(page_text.strip()) > 100:
        parts.append(f"PAGE TEXT (use for remaining text fields):\n{page_text[:30000]}")
    parts.append(f"PAGE HTML (use for any fields not covered above):\n{html_snippet}")
    if img_section:
        parts.append(f"ALL IMG TAGS FROM PAGE:\n{img_section}")

    content_section = "\n\n".join(parts)

    field_rules = []
    for f in fields:
        if f == "Name":
            field_rules.append('- "Name": the primary business name on the page.')
        elif f == "Phone":
            field_rules.append('- "Phone": any phone number. Return digits and separators only, e.g. +14155552671 or 336-517-87891.')
        elif f == "Website URL":
            field_rules.append(
                '- "Website URL": the business\'s own website URL.\n'
                '  If "WEBSITE URL" appears in the PRE-EXTRACTED FIELDS above, use that value directly.\n'
                '  Otherwise look for the business\'s own website URL (not the directory\'s domain).'
            )
        elif f == "Street":
            field_rules.append('- "Street": street address line (number + street name).')
        elif f == "City":
            field_rules.append('- "City": city name from the address.')
        elif f == "State":
            field_rules.append('- "State": state or region. Return as shown on page (abbreviation or full name).')
        elif f == "Zipcode":
            field_rules.append('- "Zipcode": postal/zip code.')
        elif f == "Country":
            field_rules.append(
                '- "Country": country name or code. '
                'Look everywhere: flag icons (🇺🇸 = US), text like "US", "United States", "UK", "United Kingdom", '
                'country abbreviations next to the address, or country fields in contact sections. '
                'A two-letter code like "US" or "GB" is a valid answer — return it as-is.'
            )
        elif f == "Category":
            field_rules.append('- "Category": the business type or industry category shown on the page.')
        elif f == "Keywords":
            field_rules.append(
                '- "Keywords": any tags, keywords, or labels associated with the business. '
                'These may appear as pipe-separated values (e.g. "AI | Legal | Law"), '
                'comma-separated tags, or a labeled section like "Keywords" or "Tags". '
                'Return comma-separated.'
            )
        elif f == "Description":
            field_rules.append(
                '- "Description": the business\'s own descriptive text about what it does.\n'
                '  If "DESCRIPTION TEXT" appears in the PRE-EXTRACTED FIELDS above, '
                'use that value directly — it is confirmed.\n'
                '  Otherwise look in PAGE TEXT for a "More about" section and extract the prose paragraph(s).\n'
                '  Do NOT return site UI text like "Payment methods accepted", "Last update:", '
                '"General information", "Write a review", or navigation/footer text.\n'
                '  Return null only if no genuine business description paragraph exists.'
            )
        elif f == "Hours":
            field_rules.append('- "Hours": business operating hours if shown, e.g. "Mon-Fri 9am-5pm".')
        elif f == "Social Media Links":
            field_rules.append(
                '- "Social Media Links": any social media URLs (Facebook, LinkedIn, Twitter/X, Instagram, WhatsApp, etc.).\n'
                '  If "SOCIAL MEDIA LINKS" appears in the PRE-EXTRACTED FIELDS above, use that value directly.\n'
                '  Otherwise search the page. Return comma-separated.'
            )
        elif f == "GBP Link":
            field_rules.append(
                '- "GBP Link": a Google Business Profile or Google Maps link.\n'
                '  If "GBP LINK" appears in the PRE-EXTRACTED FIELDS above, use that value directly.\n'
                '  Otherwise look for google.com/maps links on the page.'
            )
        elif f == "Business Email":
            field_rules.append('- "Business Email": any business email address shown on the page.')
        elif f == "Logo":
            field_rules.append(
                '- "Logo": does a business logo image exist on this page?\n'
                '  If "LOGO IMAGE FOUND IN HTML" appears in the PRE-EXTRACTED FIELDS above, '
                'return "PRESENT" immediately — it is confirmed.\n'
                '  Otherwise check ALL IMG TAGS FROM PAGE for any <img> with src containing '
                '"/logos/", "/thumb_", "logo", "profile", or "avatar".\n'
                '  Return "PRESENT" if found, null if no business logo image exists.'
            )
        elif f == "Photos":
            field_rules.append(
                '- "Photos": do business photos exist on this page?\n'
                '  Check the HTML and text for ANY of these signals:\n'
                '  * A banner image, cover photo, or hero image at the top of the listing\n'
                '  * A photo gallery, slideshow, or carousel with business images\n'
                '  * Multiple <img> tags in a gallery or photos section\n'
                '  * Any large image associated with the business listing (not site UI icons)\n'
                '  Even a single cover/banner photo counts → return "PRESENT". If none → return null.'
            )

    rules_text = "\n".join(field_rules)

    return f"""You are a business data extraction assistant. Extract business listing information from the page below.

Extract ONLY these fields: {fields}

RULES:
- Return ONLY a valid JSON object. No explanation, no markdown, no backticks.
- Search the ENTIRE page — text and HTML — for each field. Do not skip any section.
- Use null for fields genuinely not present anywhere on the page.
- Do NOT guess or invent values.
- For country: a two-letter code like "US" or "GB" shown next to an address IS the country — return it.
- For visual fields (Logo, Photos): carefully inspect ALL <img> tags in the HTML before deciding.

FIELD INSTRUCTIONS:
{rules_text}

Example output:
{{"Name": "Acme Corp", "Phone": "+14155552671", "City": "San Francisco", "State": "CA", "Country": "US", "Description": "We make great products.", "Logo": "PRESENT", "Photos": "PRESENT"}}

{content_section}"""


def _repair_truncated_json(raw: str) -> str:
    raw = raw.strip()
    if raw.endswith("}"):
        return raw
    for candidate in [raw + 'null}', raw + '"}', raw + '"}}', raw.rsplit(",", 1)[0] + "}"]:
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
            # Check if response was cut off due to token limit
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


def extract_fields(
    page_text: str, page_html: str, fields: list, source: str, api_key: str
) -> dict:
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(page_text, page_html, fields)

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

    except json.JSONDecodeError as e:
        extracted = {"_parse_error": str(e), "_raw": raw[:800], "_model": model_used}
    except Exception as e:
        extracted = {"_error": str(e), "_raw": raw[:300], "_model": model_used}

    return extracted


def extract_batch(
    scraped_pages: list, fields: list, source: str, api_key: str,
    progress_callback=None
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

            result = extract_fields(cleaned_text, cleaned_html, fields, source, api_key)

        result["_url"] = page["url"]
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, len(scraped_pages))

    return results
"""
fields_config.py
Field lists for each supported business directory source.
No layout assumptions — Gemini finds fields anywhere on the page.
"""

ALL_FIELDS = [
    "Name", "Street", "City", "State", "Zipcode", "Country",
    "Phone", "Website URL", "Keywords", "Description",
    "Hours", "Social Media Links", "GBP Link", "Business Email",
    "Category", "Logo", "Photos",
]

SOURCE_FIELDS = {
    "hotfrog.com": [
        "Name", "Street", "City", "State", "Zipcode",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Category", "Logo", "Photos",
    ],
    "brownbook.net": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Category", "Business Email", "Logo",
    ],
    "freelistingusa.com": [
        "Name",	"Street",	"City",	"State",	"Zipcode",
       "Phone",	"Website URL",	"Keywords",	"Description",	"Hours",	"Social Media Links",
        "Business Email",	"Category",	"Logo"
    ],
    "us.enrollbusiness.com":[
        "Name",	"Street",	"City",	"State",	"Zipcode",	"Country",
        "Phone",	"Website URL",	"Keywords",	"Description",	"Hours",	"Social Media Links",
        "Category",	"Logo"
    ],
    "smallbusinessusa.com":[
        "Name",	"Street",	"City",	"State",	"Zipcode", "Country",
        "Phone", "Website URL", "Category"
    ],
    "nearfinderus.com":[
        "Name",	"Street",	"City",	"State",	"Zipcode",	"Country",	"Phone",	"Website URL",
        "Description",	"Hours",	"Social Media Links",	"GBP Link",	"Business Email",	"Category",	"Logo"
    ],
    "askmap.net":[
        "Name",	"Street",	"City",	"State",	"Zipcode",
        "Phone",	"Website URL",	"Keywords",	"Description", "Hours",
        "Category",	"Logo"
    ],

    # ── New sources (default field set — adjust per-site once verified) ──────
    "callupcontact.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "zeemaps.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "place123.net": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "foursquare.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "earthmom.org": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "gravitysplash.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "webforcompany.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "provenexpert.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "thelocal.directory": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "nextbizmaker.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "surfyourtown.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "zumvu.com": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
    "blinx.biz": [
        "Name", "Street", "City", "State", "Zipcode", "Country",
        "Phone", "Website URL", "Keywords", "Description",
        "Hours", "Social Media Links", "Business Email",
        "Category", "Logo", "Photos",
    ],
}

VISUAL_FIELDS = {"Logo", "Photos"}

NA_OVERRIDES = {
    # These directories don't display a business email on the listing page,
    # so "Business Email" should be N/A rather than flagged MISSING.
    "gravitysplash.com":  ["Business Email"],
    "thelocal.directory": ["Business Email"],
}

# No site-specific layout hints — Gemini searches the whole page for each field.
SOURCE_PROMPT_HINTS = {}


def detect_source(url: str) -> str:
    """Auto-detect directory source from URL. Returns SOURCE_FIELDS key or None."""
    url_lower = url.lower()
    for source_key in SOURCE_FIELDS:
        if source_key.replace("www.", "") in url_lower:
            return source_key
    return None

import streamlit as st
from fields_config import ALL_FIELDS, SOURCE_FIELDS, VISUAL_FIELDS, detect_source
from scraper import scrape_batch
from ai_extractor import extract_batch
from comparator import compare_all
from excel_writer import write_excel, make_filename
import subprocess, sys

subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
               capture_output=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Business Listing Checker",
    page_icon="🏢",
    layout="wide",
)

st.markdown("""
<style>
    .main { max-width: 1200px; }
    .block-container { padding-top: 2rem; }
    .section-header {
        font-size: 1.1rem; font-weight: 700;
        margin-bottom: 1rem; padding-bottom: 6px;
        border-bottom: 2px solid #4472C4;
    }
    .source-badge {
        display: inline-block;
        background: #4472C4; color: white;
        border-radius: 4px; padding: 2px 8px;
        font-size: 0.78rem; font-weight: 600;
        margin: 2px 3px;
    }
    .unknown-badge {
        display: inline-block;
        background: #e74c3c; color: white;
        border-radius: 4px; padding: 2px 8px;
        font-size: 0.78rem; font-weight: 600;
        margin: 2px 3px;
    }
    .category-label {
        font-size: 0.875rem; font-weight: 600;
        margin-bottom: 0.4rem; color: inherit;
    }
    .category-hint {
        font-size: 0.78rem; color: #888;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("user_data", {}),
    ("results", None),
    ("analysis_payload", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏢 Business Listing Checker")
st.caption(
    "Enter your business data once, paste any number of directory URLs, "
    "and get a color-coded Excel report. Powered by Gemini 2.5 Flash + ScraperAPI."
)
st.markdown("---")

# ── STEP 1 — Business data form ───────────────────────────────────────────────
st.markdown('<div class="section-header">① Your Business Data</div>', unsafe_allow_html=True)
st.markdown(
    "Fill in your **correct** business information below. "
    "Only fields supported by each directory will be checked — leave others blank if you like."
)

user_data = {}

# Fields that get special treatment — excluded from the generic 3-col grid
CATEGORY_FIELD = "Category"
TEXTAREA_FIELDS = ("Description", "Keywords", "Hours", "Social Media Links", "GBP Link")

# All fields except Category go into the standard 3-column grid
fields_for_grid = [f for f in ALL_FIELDS if f != CATEGORY_FIELD]

COLS = 3
chunks = [fields_for_grid[i:i + COLS] for i in range(0, len(fields_for_grid), COLS)]

for chunk in chunks:
    cols = st.columns(COLS)
    for i, field in enumerate(chunk):
        with cols[i]:
            if field in VISUAL_FIELDS:
                sel = st.selectbox(
                    field,
                    options=["Yes — should be present", "No — not required"],
                    key=f"field_{field}",
                )
                user_data[field] = "present" if "Yes" in sel else ""

            elif field in TEXTAREA_FIELDS:
                user_data[field] = st.text_area(
                    field,
                    height=80,
                    key=f"field_{field}",
                    value=st.session_state.user_data.get(field, ""),
                )

            else:
                user_data[field] = st.text_input(
                    field,
                    key=f"field_{field}",
                    value=st.session_state.user_data.get(field, ""),
                )

# ── Category — full-width 4-column row ───────────────────────────────────────
if CATEGORY_FIELD in ALL_FIELDS:
    st.markdown(
        '<div class="category-label">Category</div>'
        '<div class="category-hint">Enter up to 4 categories — comparison passes if any one matches</div>',
        unsafe_allow_html=True,
    )
    prev_cat = st.session_state.user_data.get(CATEGORY_FIELD, "")
    prev_parts = [p.strip() for p in prev_cat.split("|")] + ["", "", "", ""]

    cat_cols = st.columns(4)
    cat_vals = []
    placeholders = ["e.g. Plumber", "e.g. Contractor", "e.g. Home Services", "e.g. Renovation"]
    for ci, col in enumerate(cat_cols):
        with col:
            cat_vals.append(
                st.text_input(
                    f"Category {ci + 1}",
                    key=f"field_Category_{ci}",
                    value=prev_parts[ci] if ci < len(prev_parts) else "",
                    placeholder=placeholders[ci],
                )
            )
    user_data[CATEGORY_FIELD] = " | ".join(v.strip() for v in cat_vals if v.strip())

st.session_state.user_data = user_data
st.markdown("---")

# ── STEP 2 — API keys ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">② API Keys</div>', unsafe_allow_html=True)
col_gem, col_scraper = st.columns(2)
with col_gem:
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Free at aistudio.google.com",
    )
with col_scraper:
    scraper_api_key = st.text_input(
        "ScraperAPI Key",
        type="password",
        placeholder="your_scraperapi_key...",
        help="Free tier at scraperapi.com (1,000 credits/month)",
    )
st.markdown("---")

# ── STEP 3 — URLs ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">③ Live Directory URLs</div>', unsafe_allow_html=True)

sorted_domains = sorted(SOURCE_FIELDS.keys())
domain_badges_html = " ".join(
    f'<span class="source-badge">{d}</span>' for d in sorted_domains
)
with st.expander(f"📋 View {len(sorted_domains)} supported directories"):
    st.markdown(domain_badges_html, unsafe_allow_html=True)

st.markdown("Paste one URL per line. URLs from **unknown** directories will be skipped.")

links_text = st.text_area(
    "Live URLs (one per line)",
    height=220,
    placeholder=(
        "https://www.hotfrog.com/company/abc123\n"
        "https://www.brownbook.net/business/xyz\n"
        "https://www.yelp.com/biz/my-business\n"
        "..."
    ),
)

raw_links = [l.strip() for l in links_text.split("\n") if l.strip().startswith("http")]

known_urls     = []
unknown_urls   = []
url_source_map = {}

for url in raw_links:
    src = detect_source(url)
    if src:
        known_urls.append(url)
        url_source_map[url] = src
    else:
        unknown_urls.append(url)

if raw_links:
    summary_parts = [f"**{len(known_urls)} recognised**"]
    if unknown_urls:
        summary_parts.append(f"**{len(unknown_urls)} unknown**")
    st.markdown(f"{len(raw_links)} URL(s) detected — " + ", ".join(summary_parts))

    with st.expander(f"Show all {len(raw_links)} URL(s)"):
        if known_urls:
            st.markdown("**✅ Recognised URLs**")
            for u in known_urls:
                src_label = url_source_map[u]
                st.markdown(
                    f'<span class="source-badge">{src_label}</span> {u}',
                    unsafe_allow_html=True,
                )
        if unknown_urls:
            st.markdown("**⚠️ Unrecognised URLs — will be skipped**")
            for u in unknown_urls:
                st.markdown(
                    f'<span class="unknown-badge">unknown</span> {u}',
                    unsafe_allow_html=True,
                )
else:
    st.warning("No valid URLs detected yet. Paste links above (must start with http).")

st.markdown("---")

# ── STEP 4 — Run ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">④ Run Analysis</div>', unsafe_allow_html=True)

run_disabled = not (known_urls and gemini_api_key and scraper_api_key)

if st.button("Start Analysis", disabled=run_disabled, type="primary"):
    # Validate inputs
    error_msg = None
    if not gemini_api_key:
        error_msg = "Please enter your Gemini API key."
    elif not scraper_api_key:
        error_msg = "Please enter your ScraperAPI key."
    elif not known_urls:
        error_msg = "No recognised directory URLs found."
    elif not [f for f in ALL_FIELDS if user_data.get(f, "").strip()]:
        error_msg = "Please fill in at least one business data field."

    if error_msg:
        st.error(error_msg)
    else:
        # Store everything needed by the analysis page
        st.session_state.analysis_payload = {
            "user_data":      user_data,
            "known_urls":     known_urls,
            "url_source_map": url_source_map,
            "gemini_api_key":  gemini_api_key,
            "scraper_api_key": scraper_api_key,
        }
        st.switch_page("pages/analysis.py")

elif run_disabled:
    hints = []
    if not gemini_api_key:  hints.append("enter your Gemini API key")
    if not scraper_api_key: hints.append("enter your ScraperAPI key")
    if not known_urls:      hints.append("paste at least one recognised directory URL")
    st.caption(f"Please {' and '.join(hints)} to enable analysis.")

# ── Re-download previous results ──────────────────────────────────────────────
if st.session_state.get("results"):
    st.markdown("---")
    st.markdown("**Previous results still available:**")
    excel_bytes = write_excel(st.session_state.results)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.download_button(
            label="📥 Re-download Last Report",
            data=excel_bytes,
            file_name=make_filename(st.session_state.user_data.get("Name", "")),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        if st.button("📊 View Results Page"):
            st.switch_page("pages/analysis.py")

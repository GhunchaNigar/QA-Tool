"""
excel_writer.py
Generates a colored Excel report from comparison results.
Supports mixed-source reports — all ALL_FIELDS as columns,
N/A shown in gray for fields not applicable to a given source.
"""

import io
import re
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from fields_config import ALL_FIELDS

# ── Color fills ───────────────────────────────────────────────────────────────
FILL_RED           = PatternFill("solid", fgColor="FF0000")
FILL_GRAY          = PatternFill("solid", fgColor="F2F2F2")
FILL_HEADER        = PatternFill("solid", fgColor="4472C4")
FILL_STATUS_HEADER = PatternFill("solid", fgColor="7030A0")

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_HEADER = Font(color="FFFFFF", bold=True, size=11)
FONT_RED    = Font(color="FFFFFF", bold=True, size=10)
FONT_NORMAL = Font(size=10)
FONT_GRAY   = Font(color="888888", size=10)

# ── Alignment ─────────────────────────────────────────────────────────────────
ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
ALIGN_LEFT   = Alignment(horizontal="left",   vertical="center", wrap_text=True)

# ── Borders ───────────────────────────────────────────────────────────────────
THIN         = Side(style="thin", color="CCCCCC")
BORDER_THIN  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
THICK_PURPLE = Side(style="medium", color="7030A0")
BORDER_STATUS_HEADER = Border(
    left=THICK_PURPLE, right=THICK_PURPLE,
    top=THICK_PURPLE,  bottom=THICK_PURPLE,
)


def make_filename(business_name: str) -> str:
    """
    Convert the business name into a safe filename.
    e.g. "HAQQ Legal AI" -> "HAQQ_Legal_AI_listing_report.xlsx"
    Falls back to "listing_checker_report.xlsx" if name is blank.
    """
    name = (business_name or "").strip()
    if not name:
        return "listing_checker_report.xlsx"
    safe = re.sub(r"[^\w\s\-]", "", name)
    safe = re.sub(r"\s+", "_", safe).strip("_")
    return f"{safe}_listing_report.xlsx" if safe else "listing_checker_report.xlsx"


def _style_cell(cell, fill=None, font=None, alignment=None, border=None):
    if fill:      cell.fill      = fill
    if font:      cell.font      = font
    if alignment: cell.alignment = alignment
    if border:    cell.border    = border


def write_excel(results: list) -> bytes:
    """
    Build a multi-source Excel report.
    Columns: Source | Live Link | Status | <all ALL_FIELDS>
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    # ── Header row ────────────────────────────────────────────────────────────
    headers = ["Source", "Live Link", "Status"] + ALL_FIELDS
    ws.append(headers)

    status_col_idx = headers.index("Status") + 1  # 1-based

    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(1, col_idx, header)
        if col_idx == status_col_idx:
            _style_cell(cell,
                fill=FILL_STATUS_HEADER, font=FONT_HEADER,
                alignment=ALIGN_CENTER, border=BORDER_STATUS_HEADER)
        else:
            _style_cell(cell,
                fill=FILL_HEADER, font=FONT_HEADER,
                alignment=ALIGN_CENTER, border=BORDER_THIN)

    # Fix header row height so wrapped multi-word headers never get clipped
    ws.row_dimensions[1].height = 30

    # ── Data rows ─────────────────────────────────────────────────────────────
    for result in results:
        row_values = [
            result.get("Source", ""),
            result.get("Live Link", ""),
            result.get("Status", ""),
        ]
        for field in ALL_FIELDS:
            row_values.append(result.get(field, "N/A"))

        ws.append(row_values)
        row_idx = ws.max_row

        for col_idx, value in enumerate(row_values, start=1):
            cell = ws.cell(row_idx, col_idx)
            if value in ("INCORRECT", "MISSING", "SCRAPE ERROR"):
                _style_cell(cell,
                    fill=FILL_RED, font=FONT_RED,
                    alignment=ALIGN_CENTER, border=BORDER_THIN)
            elif value == "N/A":
                _style_cell(cell,
                    fill=FILL_GRAY, font=FONT_GRAY,
                    alignment=ALIGN_CENTER, border=BORDER_THIN)
            elif value == "CORRECT":
                _style_cell(cell,
                    font=FONT_NORMAL, alignment=ALIGN_CENTER, border=BORDER_THIN)
            else:
                _style_cell(cell,
                    font=FONT_NORMAL, alignment=ALIGN_LEFT, border=BORDER_THIN)

    # ── Column widths ─────────────────────────────────────────────────────────
    # Explicit overrides for fixed columns
    explicit_widths = {"Source": 22, "Live Link": 45, "Status": 14}
    for col_idx, header in enumerate(headers, start=1):
        if header in explicit_widths:
            width = explicit_widths[header]
        else:
            # Auto-size: wide enough to show the header on one line (+ 4 padding)
            # so headers are never clipped regardless of content
            width = max(len(header) + 4, 16)
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()

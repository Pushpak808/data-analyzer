"""
analyzer/parser.py
──────────────────
Robust file ingestion for CSV, Excel, and JSON.

CSV  — auto-detects delimiter (comma, semicolon, tab, pipe) and
       tries multiple encodings (utf-8, latin-1, cp1252).
Excel — reads the first non-empty sheet; handles .xls and .xlsx.
JSON  — handles top-level array, object, or {"data": [...]} wrappers.

Post-parse:
  • strips whitespace from column names
  • drops fully-empty rows / columns
  • drops unnamed index-like columns (Unnamed: 0, index)
  • resets integer index
"""

from __future__ import annotations

import io
import json
import re

import pandas as pd


# ══════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════

def parse_file(contents: bytes, ext: str) -> pd.DataFrame:
    ext = ext.lower().lstrip(".")

    if ext == "csv":
        df = _parse_csv(contents)
    elif ext in ("xlsx", "xls"):
        df = _parse_excel(contents, ext)
    elif ext == "json":
        df = _parse_json(contents)
    else:
        raise ValueError(f"Unsupported file extension: .{ext}")

    return _clean(df)


# ══════════════════════════════════════════════════════════════
# PARSERS
# ══════════════════════════════════════════════════════════════

def _parse_csv(contents: bytes) -> pd.DataFrame:
    """Try encodings × delimiters until one parses cleanly."""
    encodings  = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    delimiters = [None, ",", ";", "\t", "|"]   # None → Python sniffer

    last_exc: Exception = ValueError("CSV parsing failed")

    for enc in encodings:
        try:
            text = contents.decode(enc)
        except UnicodeDecodeError:
            continue

        for sep in delimiters:
            try:
                kwargs: dict = dict(
                    sep=sep,
                    engine="python" if sep is None else "c",
                    low_memory=False,
                    on_bad_lines="warn",
                )
                if sep is None:
                    kwargs["sep"] = _sniff_delimiter(text)

                df = pd.read_csv(io.StringIO(text), **kwargs)

                # sanity: must have at least 1 row and 2 columns
                if df.shape[0] >= 1 and df.shape[1] >= 1:
                    return df

            except Exception as exc:
                last_exc = exc
                continue

    raise last_exc


def _sniff_delimiter(text: str) -> str:
    """Return the most likely delimiter from the first few lines."""
    import csv
    sample = "\n".join(text.splitlines()[:10])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _parse_excel(contents: bytes, ext: str) -> pd.DataFrame:
    engine = "openpyxl" if ext == "xlsx" else "xlrd"
    xl     = pd.ExcelFile(io.BytesIO(contents), engine=engine)

    # pick first non-empty sheet
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if not df.empty:
            return df

    raise ValueError("All sheets are empty")


def _parse_json(contents: bytes) -> pd.DataFrame:
    try:
        raw = json.loads(contents.decode("utf-8"))
    except UnicodeDecodeError:
        raw = json.loads(contents.decode("latin-1"))

    # array of objects
    if isinstance(raw, list):
        return pd.DataFrame(raw)

    # {"data": [...]}  or  {"results": [...]}
    if isinstance(raw, dict):
        for key in ("data", "results", "records", "rows", "items"):
            if key in raw and isinstance(raw[key], list):
                return pd.DataFrame(raw[key])

        # flat key-value object → single-row
        if all(not isinstance(v, (dict, list)) for v in raw.values()):
            return pd.DataFrame([raw])

        # nested dict of column → values  (like pandas orient='columns')
        try:
            return pd.DataFrame(raw)
        except ValueError:
            pass

    raise ValueError(
        "JSON must be an array of objects, or an object with a "
        "'data', 'results', or 'records' key containing an array."
    )


# ══════════════════════════════════════════════════════════════
# POST-PARSE CLEANING
# ══════════════════════════════════════════════════════════════

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # normalise column names
    df.columns = (
        df.columns
          .astype(str)
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
    )

    # drop fully-empty rows and columns
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)

    # drop auto-generated index columns  (Unnamed: 0, index, id-like with 0..N)
    to_drop = []
    for col in df.columns:
        lower = col.lower()
        if re.match(r"^unnamed[: _]\d+$", lower):
            to_drop.append(col)
        elif lower in ("index",) and _is_range_index(df[col]):
            to_drop.append(col)
    df.drop(columns=to_drop, inplace=True, errors="ignore")

    # cap extreme column counts (safety)
    if df.shape[1] > 300:
        df = df.iloc[:, :300]

    df.reset_index(drop=True, inplace=True)
    return df


def _is_range_index(series: pd.Series) -> bool:
    """True if the column looks like 0, 1, 2, … N-1."""
    try:
        nums = pd.to_numeric(series, errors="coerce")
        if nums.isna().any():
            return False
        expected = pd.Series(range(len(series)), dtype=float)
        return (nums.reset_index(drop=True) == expected).all()
    except Exception:
        return False
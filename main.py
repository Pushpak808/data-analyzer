"""
main.py  ─  DataSage API  v3.0
────────────────────────────────
Endpoints:
  GET  /                    → frontend
  POST /upload              → full analysis pipeline
  POST /chart               → generate a single custom chart (chart picker)
  POST /query               → natural language question via Claude API
  POST /summary             → AI-written dataset summary via Claude API
"""

from __future__ import annotations

import json
import math
import os
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from analyzer.charts    import generate_chart_data, generate_single_chart
from analyzer.features  import engineer_features
from analyzer.importance import score_columns
from analyzer.insights  import generate_insights
from analyzer.parser    import parse_file
from analyzer.stats     import compute_stats

from dotenv import load_dotenv
load_dotenv()


# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="DataSage API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ── In-memory session store ───────────────────────────────────
_SESSION: dict = {}


# ══════════════════════════════════════════════════════════════
# NUMPY SANITIZER  — converts ALL numpy/pandas scalars to native
# Python types so FastAPI's JSON encoder never chokes.
# ══════════════════════════════════════════════════════════════

def sanitize(obj):
    """Recursively convert numpy/pandas types → native Python."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    # numpy integer family
    if isinstance(obj, (np.integer,)):
        return int(obj)
    # numpy float family
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    # numpy bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    # pandas NA / NaT
    if obj is pd.NA or obj is pd.NaT:
        return None
    # plain Python float NaN/inf
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _make_session_key(filename: str, rows: int) -> str:
    return f"{filename}_{rows}"


async def _gemini(system: str, user: str, max_tokens: int = 1024) -> str:
    """Call Gemini 2.5 Flash and return text response."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    # Gemini REST API — system prompt goes as first user turn with role="user"
    # then model acknowledges, then actual user message follows.
    # Simpler: pass system as first contents entry with "user" role.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    payload = {
        "system_instruction": {
            "parts": [{"text": system}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.3,
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type":   "application/json",
            },
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {resp.text}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Gemini response format: {data}")


def _stats_summary(stats: dict, importance: dict, rows: int, filename: str) -> str:
    """Compact stats digest sent to Claude — keeps token cost low."""
    lines = [f"Dataset: {filename}, {rows} rows, {len(stats)} columns\n"]
    for col, s in stats.items():
        imp   = importance.get(col, {})
        ctype = s.get("type", "?")
        line  = f"  {col} [{ctype}, importance={imp.get('label','?')}]"
        if ctype == "numeric":
            line += (f": mean={s.get('mean')}, std={s.get('std')}, "
                     f"skew={s.get('skewness')}, outlier%={s.get('outlier_ratio')}, "
                     f"missing%={s.get('missing_rate')}")
        elif ctype == "categorical":
            line += (f": {s.get('unique_count')} unique, "
                     f"top='{s.get('dominant_value')}' ({s.get('dominant_rate')}%), "
                     f"entropy={s.get('categorical_entropy')}")
        elif ctype == "date":
            line += f": {s.get('min_date')} → {s.get('max_date')}, freq={s.get('inferred_freq')}"
        lines.append(line)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


# ── UPLOAD ────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = {"csv", "xlsx", "xls", "json"}
    ext     = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: .{ext}")

    contents = await file.read()

    try:
        df = parse_file(contents, ext)
    except Exception as exc:
        raise HTTPException(422, f"Could not parse file: {exc}")

    if df.empty:
        raise HTTPException(422, "File parsed but produced an empty DataFrame.")

    try:
        stats      = compute_stats(df)
        importance = score_columns(stats)
        charts     = generate_chart_data(df, stats)
        corr_chart = charts.get("correlation_matrix")
        insights   = generate_insights(stats, importance, corr_chart)
        features   = engineer_features(df, stats)
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")

    # Cache session
    session_key = _make_session_key(file.filename, len(df))
    _SESSION[session_key] = {"df": df, "stats": stats, "importance": importance}

    preview = df.head(10).where(pd.notnull(df.head(10)), None).to_dict(orient="records")

    n_numeric     = sum(1 for s in stats.values() if s["type"] == "numeric")
    n_categorical = sum(1 for s in stats.values() if s["type"] == "categorical")
    n_date        = sum(1 for s in stats.values() if s["type"] == "date")
    n_text        = sum(1 for s in stats.values() if s["type"] == "text")
    total_missing = sum(s.get("missing", 0) for s in stats.values())
    total_cells   = len(df) * len(df.columns)

    return sanitize({
        "filename":    file.filename,
        "rows":        len(df),
        "columns":     list(df.columns),
        "session_key": session_key,
        "type_counts": {
            "numeric": n_numeric, "categorical": n_categorical,
            "date": n_date, "text": n_text,
        },
        "quality": {
            "total_missing_cells": int(total_missing),
            "overall_missing_pct": round(float(total_missing) / total_cells * 100, 2) if total_cells else 0,
            "complete_columns":    sum(1 for s in stats.values() if s.get("missing", 0) == 0),
        },
        "stats":             stats,
        "column_importance": importance,
        "charts":            charts,
        "insights":          insights,
        "features":          features,
        "preview":           preview,
    })


# ── CUSTOM CHART PICKER ───────────────────────────────────────

class ChartRequest(BaseModel):
    session_key: str
    col_x:       str
    col_y:       str | None = None   # None = univariate
    chart_type:  str


@app.post("/chart")
async def custom_chart(req: ChartRequest):
    session = _SESSION.get(req.session_key)
    if not session:
        raise HTTPException(404, "Session expired. Please re-upload your file.")

    df    = session["df"]
    stats = session["stats"]

    if req.col_x not in df.columns:
        raise HTTPException(400, f"Column '{req.col_x}' not found.")
    if req.col_y and req.col_y not in df.columns:
        raise HTTPException(400, f"Column '{req.col_y}' not found.")

    try:
        chart_data = generate_single_chart(
            df    = df,
            stats = stats,
            col_x = req.col_x,
            col_y = req.col_y,
            chart_type = req.chart_type,
        )
    except Exception as exc:
        raise HTTPException(500, f"Chart generation failed: {exc}")

    if not chart_data:
        raise HTTPException(422, f"Cannot generate '{req.chart_type}' for this column combination.")

    return sanitize(chart_data)


# ── NATURAL LANGUAGE QUERY ────────────────────────────────────

class QueryRequest(BaseModel):
    session_key: str
    question:    str


@app.post("/query")
async def natural_language_query(req: QueryRequest):
    session = _SESSION.get(req.session_key)
    if not session:
        raise HTTPException(404, "Session expired. Please re-upload your file.")

    stats      = session["stats"]
    importance = session["importance"]
    df         = session["df"]

    digest = _stats_summary(stats, importance, len(df), "dataset")

    system = """You are DataSage, an expert data analyst assistant.
You are given a statistical summary of a dataset and a user question.
Answer concisely and specifically. Use numbers from the stats when relevant.
Format your answer in plain text. Keep it under 150 words.
Do NOT suggest the user upload data — you already have the stats."""

    user = f"""Dataset statistics:
{digest}

User question: {req.question}"""

    try:
        answer = await _gemini(system, user, max_tokens=400)
    except Exception as exc:
        raise HTTPException(502, str(exc))

    return sanitize({"answer": answer})


# ── AI SUMMARY ────────────────────────────────────────────────

class SummaryRequest(BaseModel):
    session_key: str
    filename:    str


@app.post("/summary")
async def ai_summary(req: SummaryRequest):
    session = _SESSION.get(req.session_key)
    if not session:
        raise HTTPException(404, "Session expired. Please re-upload your file.")

    stats      = session["stats"]
    importance = session["importance"]
    df         = session["df"]

    digest = _stats_summary(stats, importance, len(df), req.filename)

    system = """You are DataSage, a world-class data analyst.
Write a smart, human-readable summary of a dataset based on its statistics.
Structure your response as JSON with these exact keys:
  "headline": one punchy sentence (max 15 words) describing what this dataset is about
  "summary": 3-4 sentences covering: what the data contains, key patterns, data quality, and what's most worth investigating
  "key_findings": array of exactly 3 short strings, each a specific finding with a number (e.g. "67% of users are female")
  "watchouts": array of exactly 2 short strings about data quality issues or caveats
Output ONLY valid JSON. No markdown, no extra text."""

    user = f"Analyze this dataset:\n{digest}"

    try:
        raw = await _gemini(system, user, max_tokens=600)
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return raw text as summary
        result = {"headline": "Dataset Analysis", "summary": raw,
                  "key_findings": [], "watchouts": []}
    except Exception as exc:
        raise HTTPException(502, str(exc))

    return sanitize(result)


# ══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""
main.py
───────
DataSage API — FastAPI entry point.

POST /upload  → full analysis pipeline
GET  /        → frontend
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

from analyzer.charts   import generate_chart_data
from analyzer.features import engineer_features
from analyzer.importance import score_columns
from analyzer.insights import generate_insights
from analyzer.parser   import parse_file
from analyzer.stats    import compute_stats


# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="DataSage API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # ── validate extension ────────────────────────────────────
    allowed = {"csv", "xlsx", "xls", "json"}
    ext = file.filename.rsplit(".", 1)[-1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Allowed: {', '.join(sorted(allowed))}",
        )

    contents = await file.read()

    # ── parse ─────────────────────────────────────────────────
    try:
        df = parse_file(contents, ext)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="File parsed but produced an empty DataFrame.")

    # ── analysis pipeline ─────────────────────────────────────
    try:
        # 1. column statistics
        stats = compute_stats(df)

        # 2. importance ranking
        importance = score_columns(stats)

        # 3. chart data
        charts = generate_chart_data(df, stats)

        # 4. rule-based insights
        corr_chart = charts.get("correlation_matrix")
        insights   = generate_insights(stats, importance, corr_chart)

        # 5. feature engineering suggestions
        features = engineer_features(df, stats)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    # ── data preview (first 10 rows) ──────────────────────────
    preview_df = df.head(10)
    preview    = preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")

    # ── dataset-level summary ─────────────────────────────────
    n_numeric     = sum(1 for s in stats.values() if s["type"] == "numeric")
    n_categorical = sum(1 for s in stats.values() if s["type"] == "categorical")
    n_date        = sum(1 for s in stats.values() if s["type"] == "date")
    n_text        = sum(1 for s in stats.values() if s["type"] == "text")
    total_missing = sum(s.get("missing", 0) for s in stats.values())
    total_cells   = len(df) * len(df.columns)
    overall_missing_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0

    return {
        # metadata
        "filename":      file.filename,
        "rows":          len(df),
        "columns":       list(df.columns),

        # column type breakdown
        "type_counts": {
            "numeric":     n_numeric,
            "categorical": n_categorical,
            "date":        n_date,
            "text":        n_text,
        },

        # quality summary
        "quality": {
            "total_missing_cells": total_missing,
            "overall_missing_pct": overall_missing_pct,
            "complete_columns":    sum(1 for s in stats.values() if s.get("missing", 0) == 0),
        },

        # core analysis
        "stats":             stats,
        "column_importance": importance,
        "charts":            charts,

        # intelligence layer
        "insights":  insights,
        "features":  features,

        # preview
        "preview":   preview,
    }


# ══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
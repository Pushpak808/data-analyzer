"""
analyzer/insights.py
────────────────────
Generates ranked, human-readable insight sentences from column stats,
importance scores, and pairwise correlations.

No LLM required — pure rule-based signal extraction.
Each insight has: text, category, severity, and affected columns.

Categories:
  data_quality   — missing values, constant cols, duplicates
  distribution   — skew, outliers, normality
  correlation    — strong/moderate linear relationships
  trend          — monotonic numeric or date columns
  categorical    — dominance, high cardinality
  feature        — engineered column suggestions
"""

from __future__ import annotations

import math
from itertools import combinations


# ══════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════

def generate_insights(
    stats: dict,
    importance: dict,
    correlation_matrix: dict | None = None,
) -> list[dict]:
    """
    Returns a list of insight dicts, sorted by severity then category.
    Each dict:
      {
        "text":     str,
        "category": str,
        "severity": "high" | "medium" | "low",
        "cols":     list[str],
      }
    """
    insights: list[dict] = []

    insights += _data_quality_insights(stats)
    insights += _distribution_insights(stats)
    insights += _categorical_insights(stats)
    insights += _trend_insights(stats)
    insights += _correlation_insights(correlation_matrix, stats)
    insights += _importance_insights(importance)

    # deduplicate by text
    seen: set = set()
    unique: list = []
    for ins in insights:
        if ins["text"] not in seen:
            seen.add(ins["text"])
            unique.append(ins)

    # sort: high → medium → low, then alphabetically within severity
    order = {"high": 0, "medium": 1, "low": 2}
    unique.sort(key=lambda x: (order[x["severity"]], x["category"]))

    return unique[:40]   # cap at 40 insights


# ══════════════════════════════════════════════════════════════
# DATA QUALITY
# ══════════════════════════════════════════════════════════════

def _data_quality_insights(stats: dict) -> list[dict]:
    out = []

    heavy_missing   = []
    moderate_missing = []
    constant_cols   = []
    id_like_cols    = []

    for col, s in stats.items():
        m = s.get("missing_rate", 0)
        if m >= 50:
            heavy_missing.append((col, m))
        elif m >= 20:
            moderate_missing.append((col, m))

        # near-constant
        if s["type"] == "numeric":
            cv = s.get("cv_pct") or 0
            if cv < 0.5 and (s.get("n_valid") or 0) > 5:
                constant_cols.append(col)

        # ID-like categorical
        if s["type"] == "categorical":
            if (s.get("cardinality_ratio") or 0) > 0.9:
                id_like_cols.append(col)

    if heavy_missing:
        cols = [c for c, _ in heavy_missing]
        pcts = [f"{c} ({p:.0f}%)" for c, p in heavy_missing]
        out.append(_ins(
            f"{len(heavy_missing)} column(s) have >50% missing values: {', '.join(pcts)}. "
            f"Consider dropping or imputing them before modelling.",
            "data_quality", "high", cols
        ))

    if moderate_missing:
        cols = [c for c, _ in moderate_missing]
        pcts = [f"{c} ({p:.0f}%)" for c, p in moderate_missing]
        out.append(_ins(
            f"{len(moderate_missing)} column(s) have notable missing data: {', '.join(pcts)}.",
            "data_quality", "medium", cols
        ))

    if constant_cols:
        out.append(_ins(
            f"Column(s) {', '.join(constant_cols)} appear near-constant (CV < 0.5%). "
            f"They carry almost no signal for analysis or modelling.",
            "data_quality", "high", constant_cols
        ))

    if id_like_cols:
        out.append(_ins(
            f"Column(s) {', '.join(id_like_cols)} look like identifier fields "
            f"(almost every value is unique). They should be excluded from statistical analysis.",
            "data_quality", "medium", id_like_cols
        ))

    return out


# ══════════════════════════════════════════════════════════════
# DISTRIBUTION
# ══════════════════════════════════════════════════════════════

def _distribution_insights(stats: dict) -> list[dict]:
    out = []

    for col, s in stats.items():
        if s["type"] != "numeric":
            continue

        skew     = s.get("skewness") or 0
        kurt     = s.get("kurtosis") or 0
        out_r    = s.get("outlier_ratio") or 0
        out_iqr  = s.get("outliers_iqr") or 0
        cv       = s.get("cv_pct") or 0
        n        = s.get("n_valid") or 0

        # heavy skew
        if abs(skew) >= 2:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            out.append(_ins(
                f"'{col}' is heavily {direction} skewed (skewness = {skew:.2f}). "
                f"A log or square-root transformation may improve model performance.",
                "distribution", "high", [col]
            ))
        elif abs(skew) >= 1:
            direction = "right" if skew > 0 else "left"
            out.append(_ins(
                f"'{col}' shows moderate {direction} skew ({skew:.2f}).",
                "distribution", "medium", [col]
            ))

        # outliers
        if out_r >= 15:
            out.append(_ins(
                f"'{col}' has a very high outlier rate: {out_r:.1f}% of values "
                f"({out_iqr} rows) fall outside the IQR fence. Investigate before modelling.",
                "distribution", "high", [col]
            ))
        elif out_r >= 5:
            out.append(_ins(
                f"'{col}' contains notable outliers ({out_r:.1f}%, {out_iqr} rows).",
                "distribution", "medium", [col]
            ))

        # heavy tails
        if kurt > 5:
            out.append(_ins(
                f"'{col}' has very heavy tails (excess kurtosis = {kurt:.2f}). "
                f"Extreme values are much more common than a normal distribution would predict.",
                "distribution", "medium", [col]
            ))

        # high variability
        if cv and cv > 100:
            out.append(_ins(
                f"'{col}' has extremely high relative variability (CV = {cv:.0f}%). "
                f"The spread is larger than the mean — check for mixed populations.",
                "distribution", "medium", [col]
            ))

    return out


# ══════════════════════════════════════════════════════════════
# CATEGORICAL
# ══════════════════════════════════════════════════════════════

def _categorical_insights(stats: dict) -> list[dict]:
    out = []

    for col, s in stats.items():
        if s["type"] != "categorical":
            continue

        dom_r  = s.get("dominant_rate") or 0
        dom_v  = s.get("dominant_value", "")
        n_uniq = s.get("unique_count") or 0
        norm_e = s.get("normalised_entropy") or 0
        card   = s.get("cardinality_ratio") or 0

        # extreme dominance
        if dom_r >= 90:
            out.append(_ins(
                f"'{col}' is dominated by one value: '{dom_v}' accounts for "
                f"{dom_r:.0f}% of all rows. The column has very little discriminating power.",
                "categorical", "high", [col]
            ))
        elif dom_r >= 70:
            out.append(_ins(
                f"'{col}' is skewed — '{dom_v}' covers {dom_r:.0f}% of rows.",
                "categorical", "medium", [col]
            ))

        # binary
        if n_uniq == 2:
            vals = s.get("top_values", [])
            out.append(_ins(
                f"'{col}' is binary with values {vals}. "
                f"It can be directly used as a feature or target variable.",
                "categorical", "low", [col]
            ))

        # high cardinality (but not ID — already caught)
        if 0.5 < card <= 0.9:
            out.append(_ins(
                f"'{col}' has high cardinality ({s.get('unique_count')} unique values, "
                f"{card*100:.0f}% of rows). Consider grouping rare categories.",
                "categorical", "medium", [col]
            ))

    return out


# ══════════════════════════════════════════════════════════════
# TREND
# ══════════════════════════════════════════════════════════════

def _trend_insights(stats: dict) -> list[dict]:
    out = []
    date_cols    = [c for c, s in stats.items() if s["type"] == "date"]
    numeric_cols = [c for c, s in stats.items() if s["type"] == "numeric"]

    for col in date_cols:
        s   = stats[col]
        freq = s.get("inferred_freq")
        reg  = s.get("regularity_pct")
        span = s.get("range_days") or 0

        if freq:
            out.append(_ins(
                f"'{col}' appears to follow a {freq} cadence "
                f"spanning {span} days. Time-series analysis is applicable.",
                "trend", "medium", [col]
            ))

        if reg is not None and reg < 50:
            out.append(_ins(
                f"'{col}' has irregular intervals (only {reg:.0f}% of gaps are regular). "
                f"Resampling may be needed before time-series modelling.",
                "trend", "medium", [col]
            ))

    for col in numeric_cols:
        s = stats[col]
        if s.get("monotonic_inc"):
            out.append(_ins(
                f"'{col}' is strictly increasing across the dataset — "
                f"possibly a cumulative metric or time-indexed counter.",
                "trend", "low", [col]
            ))
        elif s.get("monotonic_dec"):
            out.append(_ins(
                f"'{col}' is strictly decreasing — check if it represents a declining measure.",
                "trend", "low", [col]
            ))

    return out


# ══════════════════════════════════════════════════════════════
# CORRELATION
# ══════════════════════════════════════════════════════════════

def _correlation_insights(
    corr_chart: dict | None,
    stats: dict,
) -> list[dict]:
    out = []
    if not corr_chart:
        return out

    labels = corr_chart.get("labels", [])
    matrix = corr_chart.get("matrix", [])

    if not labels or not matrix:
        return out

    strong   = []   # |r| >= 0.7
    moderate = []   # |r| >= 0.4

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            v = matrix[i][j]
            if v is None:
                continue
            r = float(v)
            if abs(r) >= 0.7:
                strong.append((labels[i], labels[j], r))
            elif abs(r) >= 0.4:
                moderate.append((labels[i], labels[j], r))

    for a, b, r in sorted(strong, key=lambda x: -abs(x[2]))[:5]:
        direction = "positively" if r > 0 else "negatively"
        out.append(_ins(
            f"Strong {direction} correlated pair: '{a}' and '{b}' (r = {r:.2f}). "
            f"{'High multicollinearity — consider dropping one in regression.' if abs(r) >= 0.9 else 'Strong linear relationship worth investigating.'}",
            "correlation", "high" if abs(r) >= 0.9 else "medium", [a, b]
        ))

    if moderate:
        pairs = ", ".join(f"'{a}'↔'{b}' ({r:.2f})" for a, b, r in moderate[:4])
        cols  = list({c for a, b, _ in moderate for c in (a, b)})
        out.append(_ins(
            f"Moderate correlations detected: {pairs}.",
            "correlation", "low", cols
        ))

    return out


# ══════════════════════════════════════════════════════════════
# IMPORTANCE
# ══════════════════════════════════════════════════════════════

def _importance_insights(importance: dict) -> list[dict]:
    out = []

    important = [c for c, v in importance.items() if v["label"] == "important"]
    low       = [c for c, v in importance.items() if v["label"] == "low"]

    if important:
        out.append(_ins(
            f"Top-ranked columns by information content: {', '.join(important[:5])}. "
            f"These should be prioritised in any analysis or model.",
            "feature", "medium", important[:5]
        ))

    if len(low) >= 3:
        out.append(_ins(
            f"{len(low)} columns scored low on importance: {', '.join(low[:5])}{'…' if len(low) > 5 else ''}. "
            f"Consider dropping them to reduce noise.",
            "feature", "low", low[:5]
        ))

    return out


# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════

def _ins(text: str, category: str, severity: str, cols: list) -> dict:
    return {
        "text":     text,
        "category": category,
        "severity": severity,
        "cols":     cols,
    }
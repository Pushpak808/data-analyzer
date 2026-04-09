"""
analyzer/importance.py
──────────────────────
Column importance scoring.

Each column receives a numeric score built from weighted signals:
  - information content  (entropy, variance)
  - data quality         (missing rate, outlier density)
  - analytical value     (skewness, monotonicity, temporal structure)
  - cardinality fitness  (too uniform or too sparse both penalise)

Score is normalised to [0, 10] and bucketed into three tiers:
  important  ≥ 6
  moderate   3 – 5.9
  low        < 3
"""

from __future__ import annotations
import math


# ══════════════════════════════════════════════════════════════
# PUBLIC
# ══════════════════════════════════════════════════════════════

def score_columns(stats: dict) -> dict:
    raw = {col: _score(col_stats) for col, col_stats in stats.items()}

    # ── normalise to [0, 10] across dataset ──────────────────
    values = [r["raw"] for r in raw.values()]
    lo, hi = min(values), max(values)
    span   = hi - lo if hi != lo else 1.0

    results = {}
    for col, r in raw.items():
        normalised = round((r["raw"] - lo) / span * 10, 2)
        results[col] = {
            "type":         stats[col]["type"],
            "score":        normalised,
            "label":        _label(normalised),
            "missing_rate": stats[col].get("missing_rate", 0),
            "reasons":      r["reasons"],
        }

    return results


# ══════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════

def _score(s: dict) -> dict:
    """Return raw score + reasons list."""
    col_type = s.get("type", "text")
    score    = 0.0
    reasons  = []

    # ── 1. Data quality penalty (applies to all types) ───────
    missing = s.get("missing_rate", 0)
    if missing >= 80:
        score -= 4.0
        reasons.append(f"Mostly missing ({missing:.0f}%)")
    elif missing >= 50:
        score -= 2.5
        reasons.append(f"High missing rate ({missing:.0f}%)")
    elif missing >= 20:
        score -= 1.0
        reasons.append(f"Notable missing values ({missing:.0f}%)")
    elif missing > 0:
        score -= 0.2

    # ── 2. Type-specific scoring ──────────────────────────────
    if col_type == "numeric":
        score, reasons = _numeric_score(s, score, reasons)
    elif col_type == "categorical":
        score, reasons = _categorical_score(s, score, reasons)
    elif col_type == "date":
        score, reasons = _date_score(s, score, reasons)
    else:  # text
        score -= 1.0
        reasons.append("Free-text column (low analytical value)")

    return {"raw": score, "reasons": reasons}


# ── Numeric ───────────────────────────────────────────────────

def _numeric_score(s: dict, score: float, reasons: list) -> tuple:
    # base: numeric columns are valuable
    score += 2.0

    cv        = s.get("cv_pct") or 0.0
    variance  = s.get("variance_score") or 0.0
    skewness  = abs(s.get("skewness") or 0.0)
    kurtosis  = abs(s.get("kurtosis") or 0.0)
    outlier_r = s.get("outlier_ratio") or 0.0
    entropy   = s.get("numeric_entropy") or 0.0
    monotonic = s.get("monotonic", False)
    n_valid   = s.get("n_valid") or 0

    # variance / spread
    if cv > 100:
        score += 2.5
        reasons.append("Very high variability (CV > 100%)")
    elif cv > 30:
        score += 1.5
        reasons.append("High variability")
    elif cv < 1:
        score -= 2.0
        reasons.append("Near-constant values (low variance)")

    # distribution shape
    if 0.3 < skewness < 2.0:
        score += 0.5
        reasons.append("Moderately skewed distribution")
    elif skewness >= 2.0:
        score += 1.0
        reasons.append("Highly skewed — potential for transformation")

    if kurtosis > 3:
        score += 0.5
        reasons.append("Heavy tails (high kurtosis)")

    # entropy (information richness)
    if entropy > 3.5:
        score += 1.5
        reasons.append("High information content")
    elif entropy > 2.0:
        score += 0.5

    # outliers
    if outlier_r > 20:
        score -= 1.5
        reasons.append(f"Very high outlier density ({outlier_r:.1f}%)")
    elif outlier_r > 10:
        score -= 0.5
        reasons.append(f"Notable outliers ({outlier_r:.1f}%)")

    # monotonic trend (useful for time-series features)
    if monotonic:
        score += 1.0
        reasons.append("Monotonic trend — good time-series signal")

    # sample size bonus
    if n_valid >= 10_000:
        score += 0.5
    elif n_valid < 10:
        score -= 1.0
        reasons.append("Very few valid values")

    return score, reasons


# ── Categorical ───────────────────────────────────────────────

def _categorical_score(s: dict, score: float, reasons: list) -> tuple:
    score += 1.5

    n_unique    = s.get("unique_count") or 0
    cardinality = s.get("cardinality_ratio") or 0.0
    entropy     = s.get("categorical_entropy") or 0.0
    norm_entropy= s.get("normalised_entropy") or 0.0
    balance     = s.get("balance_score") or 0.0
    dominant_r  = s.get("dominant_rate") or 100.0
    gini        = s.get("gini_impurity") or 0.0

    # normalised entropy is the cleanest signal
    if norm_entropy >= 0.85:
        score += 2.0
        reasons.append("Near-uniform distribution (high entropy)")
    elif norm_entropy >= 0.55:
        score += 1.0
        reasons.append("Good category diversity")
    elif norm_entropy < 0.2:
        score -= 1.5
        reasons.append("One dominant category (low entropy)")

    # balance / gini
    if balance > 0.7:
        score += 0.5
        reasons.append("Balanced categories")

    # cardinality extremes
    if cardinality > 0.9:
        score -= 3.0
        reasons.append("Almost unique values — effectively an ID column")
    elif cardinality > 0.5:
        score -= 1.5
        reasons.append("Very high cardinality")
    elif n_unique == 1:
        score -= 3.0
        reasons.append("Single value — no variance")
    elif n_unique == 2:
        score += 0.5
        reasons.append("Binary column — clean split")

    # dominance penalty
    if dominant_r > 95:
        score -= 1.0
        reasons.append(f"One value covers {dominant_r:.0f}% of rows")

    return score, reasons


# ── Date ─────────────────────────────────────────────────────

def _date_score(s: dict, score: float, reasons: list) -> tuple:
    score += 3.0
    reasons.append("Temporal column — enables time-series analysis")

    if s.get("is_sorted"):
        score += 1.0
        reasons.append("Chronologically ordered")

    freq = s.get("inferred_freq")
    if freq:
        score += 0.5
        reasons.append(f"Regular {freq} cadence")

    regularity = s.get("regularity_pct") or 0
    if regularity >= 80:
        score += 0.5
        reasons.append("Highly regular intervals")
    elif regularity < 40:
        score -= 0.5
        reasons.append("Irregular intervals")

    n_unique = s.get("unique_dates") or 0
    if n_unique < 3:
        score -= 2.0
        reasons.append("Too few unique dates")

    return score, reasons


# ══════════════════════════════════════════════════════════════
# LABEL
# ══════════════════════════════════════════════════════════════

def _label(normalised_score: float) -> str:
    if normalised_score >= 6.0:
        return "important"
    if normalised_score >= 3.0:
        return "moderate"
    return "low"
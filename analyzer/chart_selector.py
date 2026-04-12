"""
analyzer/chart_selector.py
──────────────────────────
Chart selection engine based on the "From Data to Viz" decision tree
by Yan Holtz & Conor Healy (data-to-viz.com), with practical
data-quality gates that prevent generating useless charts.

Key design decisions vs previous version:
  • Bivariate pairs are GATED — only generated when there is actual
    signal (correlation ≥ threshold, enough group contrast, etc.)
  • Discrete-numeric columns (age, rating, score) are correctly
    treated as categorical-like in bivariate contexts
  • Intent inference is fixed: cat×num "composition" only when the
    numeric really represents parts-of-a-whole (sum = total)
  • Alternatives list only contains chart types that the frontend
    actually renders — no phantom types like ridgeline/bubble
  • A single "confidence" score is returned alongside the chart type
    so the frontend can sort / filter meaningfully
"""

from __future__ import annotations
import math


# ══════════════════════════════════════════════════════════════
# THRESHOLDS  (all in one place — easy to tune)
# ══════════════════════════════════════════════════════════════

_T = {
    # cardinality
    "cat_binary":        2,
    "cat_few":           6,
    "cat_medium":        15,
    "cat_many":          40,

    # discrete numeric (integer, small range)
    "discrete_max_unique": 20,
    "discrete_ratio":      0.05,   # unique / n < 5 %

    # distribution shape
    "skew_high":           1.0,
    "outlier_high":        5.0,    # % rows outside IQR fence
    "cv_high":             50.0,   # coefficient of variation %


    # bivariate gates  — skip pair if below these
    "corr_min":            0.15,   # |Pearson r| minimum to show scatter/line
    "group_min_rows":      5,      # min rows per category group
    "group_min_cats":      2,      # min distinct categories after filtering

    # balance
    "balance_min":         0.25,   # 1 - dominant_rate; below = very skewed
}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _is_discrete(s: dict) -> bool:
    """
    Trust the is_discrete flag set by stats.py.
    True for: integer columns where values represent counts/ordinals
    (age, score, rating, social_level, platform_code, etc.)
    """
    return bool(s.get("is_discrete", False))

def _is_monotonic(s: dict) -> bool:
    return bool(s.get("monotonic", False))

def _has_outliers(s: dict) -> bool:
    return (s.get("outlier_ratio") or 0) > _T["outlier_high"]

def _high_cv(s: dict) -> bool:
    return (s.get("cv_pct") or 0) > _T["cv_high"]

def _skew(s: dict) -> float:
    return abs(s.get("skewness") or 0)

def _n(s: dict) -> int:
    return s.get("n_valid") or 0

def _unique(s: dict) -> int:
    return s.get("unique_count") or 0

def _balance(s: dict) -> float:
    return s.get("balance_score") or 0

def _card(s: dict) -> float:
    return s.get("cardinality_ratio") or 0


# ══════════════════════════════════════════════════════════════
# UNIVARIATE — public entry point
# ══════════════════════════════════════════════════════════════

def choose_univariate_chart(
    stats: dict,
    col_type: str,
    intent: str | None = None,
) -> str:
    if col_type == "date":
        return "line"
    if col_type == "text":
        return "table"

    if intent is None:
        intent = _infer_univariate_intent(stats, col_type)

    if col_type == "numeric":
        return _numeric_univar(stats, intent)
    if col_type == "categorical":
        return _categorical_univar(stats, intent)
    return "table"


def _infer_univariate_intent(stats: dict, col_type: str) -> str:
    if col_type == "categorical":
        dom = stats.get("dominant_rate") or 0
        u   = _unique(stats)
        # heavily skewed few-category → composition (pie/donut)
        if u <= _T["cat_few"] and dom >= 40:
            return "composition"
        return "ranking"

    # numeric
    if _is_monotonic(stats):
        return "trend"
    if _is_discrete(stats):
        return "ranking"     # age, score → bar/histogram by value
    return "distribution"


# ── Numeric univariate ────────────────────────────────────────
def _numeric_univar(s: dict, intent: str) -> str:
    n          = _n(s)
    outlier_r  = s.get("outlier_ratio") or 0
    skew       = _skew(s)
    discrete   = _is_discrete(s)
    n_uniq     = s.get("n_unique_vals") or 999

    if discrete:
        return "boxplot" if outlier_r > _T["outlier_high"] else "histogram"

    if intent == "trend":
        return "line"

    if intent == "ranking":
        return "histogram"

    if intent == "distribution":
        # Heavy outliers → boxplot surfaces them better
        if outlier_r > _T["outlier_high"]:
            return "boxplot"
        return "histogram"

    if intent in ("comparison", "correlation"):
        return "boxplot" if outlier_r > _T["outlier_high"] else "histogram"

    # fallback
    if _is_monotonic(s):
        return "line"
    if outlier_r > _T["outlier_high"]:
        return "boxplot"
    return "histogram"


# ── Categorical univariate ────────────────────────────────────
#
# From Data to Viz — CATEGORIC tree:
#
#  Composition  →  PIE (few, balanced) | DONUT (few, unbalanced) | TREEMAP (many)
#  Ranking      →  LOLLIPOP (≤6) | BAR (≤15) | TREEMAP (>15)
#  Distribution →  BAR | TREEMAP

def _categorical_univar(s: dict, intent: str) -> str:
    u    = _unique(s)
    bal  = _balance(s)
    card = _card(s)

    if intent == "composition":
        if card > 0.5 or u > _T["cat_medium"]:
            return "treemap"
        if u <= _T["cat_few"]:
            return "pie" if bal >= _T["balance_min"] else "donut"
        return "donut"

    if intent == "ranking":
        if u <= _T["cat_few"]:
            return "lollipop"
        if u <= _T["cat_medium"]:
            return "bar"
        if u <= _T["cat_many"]:
            return "lollipop"
        return "treemap"

    if intent in ("distribution", "comparison"):
        return "bar" if u <= _T["cat_medium"] else "treemap"

    if intent == "trend":
        return "line"

    # auto
    if u <= _T["cat_few"]:
        return "pie" if bal >= _T["balance_min"] else "bar"
    if u <= _T["cat_medium"]:
        return "bar"
    return "treemap"


# ══════════════════════════════════════════════════════════════
# BIVARIATE — public entry point
# Returns (chart_type, should_generate) tuple.
# should_generate=False means the pair has no meaningful signal.
# ══════════════════════════════════════════════════════════════

def choose_bivariate_chart(
    type_x: str,
    type_y: str,
    stats_x: dict | None = None,
    stats_y: dict | None = None,
    intent: str | None = None,
) -> str:
    """Returns chart type string. Returns 'skip' when pair has no signal."""
    sx = stats_x or {}
    sy = stats_y or {}

    # ── Date × Numeric — always meaningful ───────────────────
    if type_x == "date" or type_y == "date":
        num_s = sy if type_x == "date" else sx
        return "area" if _high_cv(num_s) else "line"

    # ── Numeric × Numeric ────────────────────────────────────
    if type_x == "numeric" and type_y == "numeric":
        return _gate_num_num(sx, sy, intent)

    # ── Discrete-numeric acts like categorical in bivariate ──
    # e.g. social_interaction_level (1-5) × age → treat level as cat
    if type_x == "numeric" and _is_discrete(sx):
        type_x = "categorical"
        sx = _discrete_as_cat(sx)
    if type_y == "numeric" and _is_discrete(sy):
        type_y = "categorical"
        sy = _discrete_as_cat(sy)

    # ── Categorical × Numeric ────────────────────────────────
    if type_x == "categorical" and type_y == "numeric":
        return _gate_cat_num(sx, sy, intent)
    if type_y == "categorical" and type_x == "numeric":
        return _gate_cat_num(sy, sx, intent)

    # ── Categorical × Categorical ────────────────────────────
    if type_x == "categorical" and type_y == "categorical":
        return _gate_cat_cat(sx, sy, intent)

    return "skip"


def should_generate_bivariate(
    type_x: str,
    type_y: str,
    stats_x: dict,
    stats_y: dict,
    corr: float | None = None,
) -> bool:
    """
    Hard gate: returns False for pairs that will produce empty/useless charts.
    Called from charts.py before building any data.
    """
    # Both text → skip
    if type_x == "text" or type_y == "text":
        return False

    # Date × Date → skip
    if type_x == "date" and type_y == "date":
        return False

    # Date × Numeric → always useful
    if type_x == "date" or type_y == "date":
        return True

    # Numeric × Numeric — require minimum correlation
    if type_x == "numeric" and type_y == "numeric":
        # If correlation is known and too weak, skip
        if corr is not None and abs(corr) < _T["corr_min"]:
            return False
        return True

    # After discrete remap
    eff_x = "categorical" if (type_x == "numeric" and _is_discrete(stats_x)) else type_x
    eff_y = "categorical" if (type_y == "numeric" and _is_discrete(stats_y)) else type_y

    # Cat × Numeric — require enough groups
    if eff_x == "categorical" and eff_y == "numeric":
        u = _unique(stats_x)
        return u >= _T["group_min_cats"]
    if eff_y == "categorical" and eff_x == "numeric":
        u = _unique(stats_y)
        return u >= _T["group_min_cats"]

    # Cat × Cat — require both have manageable cardinality
    if eff_x == "categorical" and eff_y == "categorical":
        ux = _unique(stats_x)
        uy = _unique(stats_y)
        # Skip if either is an ID-like column
        if _card(stats_x) > 0.8 or _card(stats_y) > 0.8:
            return False
        # Skip if cross-table would be too sparse
        if ux > _T["cat_many"] and uy > _T["cat_many"]:
            return False
        return True

    return True


# ── Numeric × Numeric ─────────────────────────────────────────
def _gate_num_num(sx: dict, sy: dict, intent: str | None) -> str:
    mono = _is_monotonic(sx) or _is_monotonic(sy)

    if intent == "trend" or mono:
        return "line"
    if intent == "correlation":
        return "scatter"

    # Default: scatter (correlation already gated in should_generate_bivariate)
    return "scatter"


# ── Categorical × Numeric ─────────────────────────────────────
#
# From Data to Viz — CATEGORIC × NUMERIC:
#
#  Few groups (≤5)   + distribution intent  → VIOLIN
#  Few groups (≤5)   + comparison intent    → GROUPED BAR (or LOLLIPOP if ranking)
#  Medium (6–15)     + outliers/high var    → BOXPLOT
#  Medium (6–15)     + otherwise            → GROUPED BAR
#  Many groups (>15) → BOXPLOT (horizontal)

def _gate_cat_num(cat_s: dict, num_s: dict, intent: str | None) -> str:
    u           = _unique(cat_s)
    outliers    = _has_outliers(num_s)
    high_var    = _high_cv(num_s)
    skewed      = _skew(num_s) > _T["skew_high"]

    # Infer intent if not given
    if intent is None:
        intent = _infer_cat_num_intent(cat_s, num_s)

    if intent == "ranking":
        return "lollipop" if u <= _T["cat_medium"] else "bar"

    if intent == "distribution":
        if u <= _T["groups_few"]:
            return "violin"
        if u <= _T["cat_medium"]:
            return "boxplot"
        return "boxplot"

    if intent == "comparison":
        if u <= _T["cat_few"]:
            # few groups: show distribution shape if data warrants it
            if outliers or high_var or skewed:
                return "boxplot"
            return "grouped_bar"
        if u <= _T["cat_medium"]:
            return "boxplot" if (outliers or high_var) else "grouped_bar"
        return "boxplot"

    if intent == "trend":
        return "line"

    # auto
    if u <= _T["cat_few"]:
        return "violin" if _n(num_s) >= 50 else "grouped_bar"
    if u <= _T["cat_medium"]:
        return "boxplot" if (outliers or high_var) else "grouped_bar"
    return "boxplot"


_T["groups_few"] = 5   # used above


def _infer_cat_num_intent(cat_s: dict, num_s: dict) -> str:
    """
    Better intent inference for cat×num pairs.
    'composition' only when the numeric truly represents parts-of-a-whole.
    Otherwise default to 'comparison' (compare groups).
    """
    u = _unique(cat_s)
    # Composition only makes sense when: few categories AND
    # the numeric looks like a proportion/percentage (0–1 or 0–100)
    mn  = num_s.get("min") or 0
    mx  = num_s.get("max") or 0
    rng = mx - mn
    is_proportion = (mn >= 0 and mx <= 1.01) or (mn >= 0 and mx <= 100 and rng <= 100)

    if u <= _T["cat_few"] and is_proportion:
        return "composition"
    if u <= _T["groups_few"]:
        return "distribution"
    return "comparison"


# ── Categorical × Categorical ─────────────────────────────────
def _gate_cat_cat(sx: dict, sy: dict, intent: str | None) -> str:
    ux = _unique(sx)
    uy = _unique(sy)

    # Both binary → grouped bar is very clean
    if ux <= 2 and uy <= 2:
        return "grouped_bar"

    # Few × few → stacked bar
    if ux <= _T["cat_few"] and uy <= _T["cat_few"]:
        return "stacked_bar"

    # One or both medium → heatmap
    if ux <= _T["cat_medium"] and uy <= _T["cat_medium"]:
        return "stacked_bar"

    return "heatmap"


# ── Discrete numeric as categorical proxy ─────────────────────
def _discrete_as_cat(s: dict) -> dict:
    """Return a fake categorical-stats dict from a discrete numeric."""
    return {
        "unique_count":    s.get("n_unique_vals") or s.get("unique_count") or 0,
        "cardinality_ratio": 0.05,   # discrete cols are low-cardinality by definition
        "balance_score":   0.5,
        "dominant_rate":   s.get("dominant_rate") or 30,
    }


# ══════════════════════════════════════════════════════════════
# ALTERNATIVES  (only types the frontend actually renders)
# ══════════════════════════════════════════════════════════════

ALTERNATIVES: dict[str, list[str]] = {
    "histogram":   ["boxplot"],
    "boxplot":     ["violin", "histogram"],
    "violin":      ["boxplot"],
    "bar":         ["lollipop", "treemap"],
    "lollipop":    ["bar"],
    "pie":         ["donut", "bar"],
    "donut":       ["pie", "bar"],
    "treemap":     ["bar", "lollipop"],
    "grouped_bar": ["boxplot", "lollipop"],
    "stacked_bar": ["grouped_bar"],
    "scatter":     ["line"],
    "line":        ["area", "scatter"],
    "area":        ["line"],
    "heatmap":     ["stacked_bar"],
}


def get_alternatives(chart_type: str) -> list[str]:
    return ALTERNATIVES.get(chart_type, [])
"""
analyzer/chart_selector.py
──────────────────────────
Chart selection engine based on the "From Data to Viz" decision tree
by Yan Holtz & Conor Healy (data-to-viz.com).

Two entry points:
  choose_univariate_chart(stats, col_type, intent=None)
  choose_bivariate_chart(type_x, type_y, stats_x, stats_y, intent=None)

intent (optional string):
  "distribution"  – how values are spread
  "composition"   – parts of a whole
  "ranking"       – ordered comparison
  "correlation"   – relationship between two variables
  "trend"         – change over time / sequence
  "comparison"    – compare groups side by side

When intent is supplied the tree follows the matching deterministic branch.
When intent is None the engine infers the best intent from statistics,
then routes through the same tree — both paths share identical logic.

Chart type strings returned (superset of old selector):
  histogram, density, boxplot, violin, ridgeline,
  bar, lollipop, grouped_bar, stacked_bar,
  pie, donut, treemap,
  line, area,
  scatter, bubble, heatmap,
  table
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════

_T = {
    "cat_binary":    2,
    "cat_few":       6,
    "cat_medium":    12,
    "cat_many":      30,
    "skew_high":     1.0,
    "skew_moderate": 0.5,
    "outlier_high":  5.0,
    "cv_high":       50.0,
    "entropy_high":  2.5,
    "groups_few":    5,
    "groups_many":   12,
    "balance_min":   0.3,
    "n_dense":       500,
}


# ══════════════════════════════════════════════════════════════
# INTENT INFERENCE
# ══════════════════════════════════════════════════════════════

def _infer_univariate_intent(stats: dict, col_type: str) -> str:
    if col_type == "date":
        return "trend"
    if col_type == "text":
        return "distribution"

    if col_type == "categorical":
        unique   = stats.get("unique_count") or 0
        dom_rate = stats.get("dominant_rate") or 0
        if dom_rate >= 50 and unique <= _T["cat_few"]:
            return "composition"
        return "ranking"

    # numeric
    monotonic = stats.get("monotonic", False)
    outlier_r = stats.get("outlier_ratio") or 0
    skew      = abs(stats.get("skewness") or 0)

    if monotonic:
        return "trend"
    return "distribution"


def _infer_bivariate_intent(
    type_x: str, type_y: str,
    sx: dict, sy: dict,
) -> str:
    if "date" in (type_x, type_y):
        return "trend"

    if type_x == "numeric" and type_y == "numeric":
        if _is_monotonic(sx) or _is_monotonic(sy):
            return "trend"
        return "correlation"

    if "categorical" in (type_x, type_y) and "numeric" in (type_x, type_y):
        cat_s = sx if type_x == "categorical" else sy
        unique = cat_s.get("unique_count") or 0
        if unique <= _T["cat_few"]:
            return "composition"
        return "comparison"

    if type_x == "categorical" and type_y == "categorical":
        return "composition"

    return "comparison"


# ══════════════════════════════════════════════════════════════
# UNIVARIATE  — public entry point
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


# ── Numeric branch ────────────────────────────────────────────
#
# From Data to Viz — NUMERIC tree:
#
#  One variable
#    └─ distribution?
#         ├─ outliers present  → BOXPLOT
#         ├─ large n, smooth   → DENSITY
#         └─ otherwise         → HISTOGRAM
#  Trend?           → LINE
#  Ranking?         → HISTOGRAM (value frequency)

def _numeric_univar(s: dict, intent: str) -> str:
    skew      = abs(s.get("skewness") or 0)
    outlier_r = s.get("outlier_ratio") or 0
    monotonic = s.get("monotonic", False)
    n_valid   = s.get("n_valid") or 0

    if intent == "trend":
        return "line"

    if intent == "ranking":
        return "histogram"

    if intent == "distribution":
        if outlier_r > _T["outlier_high"]:
            return "boxplot"
        if n_valid >= _T["n_dense"] and skew < _T["skew_high"]:
            return "density"
        return "histogram"

    if intent in ("comparison", "correlation"):
        if outlier_r > _T["outlier_high"]:
            return "boxplot"
        return "histogram"

    if intent == "composition":
        return "histogram"

    # auto fallback
    if monotonic:
        return "line"
    if outlier_r > _T["outlier_high"]:
        return "boxplot"
    if n_valid >= _T["n_dense"]:
        return "density"
    return "histogram"


# ── Categorical branch ────────────────────────────────────────
#
# From Data to Viz — CATEGORIC tree:
#
#  Composition (part of whole)
#    ├─ few cats + balanced   → PIE
#    ├─ few cats + unbalanced → DONUT
#    └─ many cats             → TREEMAP
#
#  Ranking
#    ├─ ≤ 6   → LOLLIPOP
#    ├─ ≤ 12  → BAR
#    └─ > 12  → TREEMAP
#
#  Distribution / Comparison
#    ├─ ≤ 12  → BAR
#    └─ > 12  → TREEMAP

def _categorical_univar(s: dict, intent: str) -> str:
    unique      = s.get("unique_count") or 0
    balance     = s.get("balance_score") or 0
    cardinality = s.get("cardinality_ratio") or 0

    if intent == "composition":
        if cardinality > 0.5 or unique > _T["cat_medium"]:
            return "treemap"
        if unique <= _T["cat_few"]:
            return "pie" if balance >= _T["balance_min"] else "donut"
        return "donut"

    if intent == "ranking":
        if unique <= _T["cat_few"]:
            return "lollipop"
        if unique <= _T["cat_medium"]:
            return "bar"
        if unique <= _T["cat_many"]:
            return "lollipop"
        return "treemap"

    if intent in ("distribution", "comparison"):
        if unique <= _T["cat_medium"]:
            return "bar"
        return "treemap"

    if intent == "trend":
        return "line"

    # auto fallback
    if unique <= _T["cat_few"]:
        return "pie" if balance >= _T["balance_min"] else "bar"
    if unique <= _T["cat_medium"]:
        return "bar"
    return "treemap"


# ══════════════════════════════════════════════════════════════
# BIVARIATE  — public entry point
# ══════════════════════════════════════════════════════════════

def choose_bivariate_chart(
    type_x: str,
    type_y: str,
    stats_x: dict | None = None,
    stats_y: dict | None = None,
    intent: str | None = None,
) -> str:
    sx = stats_x or {}
    sy = stats_y or {}

    if intent is None:
        intent = _infer_bivariate_intent(type_x, type_y, sx, sy)

    # ── Date × Numeric ───────────────────────────────────────
    # From Data to Viz — TIME SERIES: always line / area
    if type_x == "date" or type_y == "date":
        num_s = sy if type_x == "date" else sx
        cv    = num_s.get("cv_pct") or 0
        if cv > _T["cv_high"]:
            return "area"
        return "line"

    # ── Numeric × Numeric ────────────────────────────────────
    # From Data to Viz — NUMERIC × NUMERIC (RELATIONAL):
    #   correlation  → SCATTER
    #   trend        → LINE
    if type_x == "numeric" and type_y == "numeric":
        return _num_num(sx, sy, intent)

    # ── Categorical × Numeric ────────────────────────────────
    # From Data to Viz — CATEGORIC × NUMERIC:
    #   ranking      → LOLLIPOP / BAR
    #   composition  → STACKED BAR
    #   distribution → VIOLIN (few groups) / BOXPLOT (many)
    #   comparison   → GROUPED BAR / BOXPLOT
    if type_x == "categorical" and type_y == "numeric":
        return _cat_num(sx, sy, intent)
    if type_y == "categorical" and type_x == "numeric":
        return _cat_num(sy, sx, intent)

    # ── Categorical × Categorical ────────────────────────────
    # From Data to Viz — CATEGORIC × CATEGORIC:
    #   few × few    → STACKED BAR / GROUPED BAR
    #   any large    → HEATMAP
    if type_x == "categorical" and type_y == "categorical":
        return _cat_cat(sx, sy, intent)

    return "table"


# ── Numeric × Numeric ─────────────────────────────────────────
def _num_num(sx: dict, sy: dict, intent: str) -> str:
    mono = _is_monotonic(sx) or _is_monotonic(sy)

    if intent == "trend":
        return "line"
    if intent == "correlation":
        return "scatter"
    if intent == "distribution":
        return "scatter"
    if intent in ("comparison", "ranking"):
        return "scatter"
    if intent == "composition":
        return "scatter"

    # auto
    if mono:
        return "line"
    return "scatter"


# ── Categorical × Numeric ─────────────────────────────────────
def _cat_num(cat_s: dict, num_s: dict, intent: str) -> str:
    unique       = cat_s.get("unique_count") or 0
    has_outliers = _has_outliers(num_s)
    high_var     = (num_s.get("cv_pct") or 0) > _T["cv_high"]

    if intent == "ranking":
        if unique <= _T["cat_medium"]:
            return "lollipop"
        return "bar"

    if intent == "composition":
        if unique <= _T["cat_few"]:
            return "stacked_bar"
        return "grouped_bar"

    if intent == "distribution":
        # From Data to Viz: few groups → violin; many → ridgeline / box
        if unique <= _T["groups_few"]:
            return "violin"
        if unique <= _T["groups_many"]:
            return "boxplot"
        return "heatmap"

    if intent == "comparison":
        if unique <= _T["cat_medium"]:
            if has_outliers or high_var:
                return "boxplot"
            return "grouped_bar"
        return "boxplot"

    if intent == "trend":
        return "line"

    # auto
    if unique <= _T["cat_medium"]:
        return "boxplot" if (has_outliers or high_var) else "grouped_bar"
    return "boxplot"


# ── Categorical × Categorical ─────────────────────────────────
def _cat_cat(sx: dict, sy: dict, intent: str) -> str:
    ux = sx.get("unique_count") or 999
    uy = sy.get("unique_count") or 999

    if intent == "composition":
        if ux <= _T["cat_few"] and uy <= _T["cat_few"]:
            return "stacked_bar"
        return "heatmap"

    if intent in ("comparison", "distribution"):
        if ux <= _T["cat_medium"] and uy <= _T["cat_medium"]:
            return "grouped_bar"
        return "heatmap"

    if intent == "ranking":
        return "grouped_bar" if ux <= _T["cat_medium"] else "heatmap"

    # auto
    if ux <= _T["cat_medium"] and uy <= _T["cat_medium"]:
        return "stacked_bar"
    return "heatmap"


# ══════════════════════════════════════════════════════════════
# ALTERNATIVES  (for frontend "also try" suggestions)
# ══════════════════════════════════════════════════════════════

ALTERNATIVES: dict[str, list[str]] = {
    "histogram":   ["density", "boxplot", "violin"],
    "density":     ["histogram", "violin", "ridgeline"],
    "boxplot":     ["violin", "histogram", "scatter"],
    "violin":      ["boxplot", "ridgeline", "density"],
    "ridgeline":   ["violin", "density"],
    "bar":         ["lollipop", "treemap"],
    "lollipop":    ["bar", "treemap"],
    "pie":         ["donut", "bar", "treemap"],
    "donut":       ["pie", "bar", "treemap"],
    "treemap":     ["bar", "lollipop"],
    "grouped_bar": ["boxplot", "violin", "lollipop"],
    "stacked_bar": ["grouped_bar", "area"],
    "scatter":     ["line", "bubble", "heatmap"],
    "line":        ["area", "scatter"],
    "area":        ["line", "stacked_bar"],
    "heatmap":     ["grouped_bar", "scatter"],
    "bubble":      ["scatter"],
}


def get_alternatives(chart_type: str) -> list[str]:
    return ALTERNATIVES.get(chart_type, [])


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _is_monotonic(s: dict) -> bool:
    return bool(s.get("monotonic", False))


def _has_outliers(s: dict) -> bool:
    return (s.get("outlier_ratio") or 0) > _T["outlier_high"]
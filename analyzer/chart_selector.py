"""
analyzer/chart_selector.py
──────────────────────────
Recommends the best chart type for a given column or column pair,
using a weighted scoring engine driven by column statistics.

Each chart type accumulates evidence points; the highest scorer wins.
Weights and thresholds are centralised at the top for easy tuning.
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════
# WEIGHTS  (evidence points per signal)
# ══════════════════════════════════════════════════════════════

_W = {
    # histogram
    "hist_base":          2,
    "hist_skew":          2,
    "hist_entropy":       1,
    "hist_variance":      1,
    "hist_many_vals":     1,

    # box
    "box_outliers":       4,
    "box_skew":           1,
    "box_variance":       1,

    # bar
    "bar_base":           2,
    "bar_few_cats":       2,
    "bar_entropy":        1,

    # pie
    "pie_small":          2,
    "pie_balance":        1,
    "pie_binary":         2,
    "pie_penalty_large":  -3,

    # line
    "line_time":          5,
    "line_monotonic":     3,

    # scatter
    "scatter_base":       3,
    "scatter_entropy":    1,

    # stacked bar
    "stacked_base":       3,

    # grouped bar / box
    "grouped_bar_few":    3,
    "grouped_box_many":   3,
    "grouped_box_var":    1,
}

# ══════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════

_T = {
    "skew_high":            1.0,
    "entropy_numeric_high": 2.5,
    "entropy_cat_high":     1.5,
    "outlier_pct_high":     5.0,
    "variance_high":        1.0,
    "cv_high":              50.0,
    "pie_max_unique":       6,
    "bar_few_unique":       15,
    "cat_many_unique":      12,
}


# ══════════════════════════════════════════════════════════════
# UNIVARIATE
# ══════════════════════════════════════════════════════════════

def choose_univariate_chart(stats: dict, col_type: str) -> str:
    if col_type == "numeric":
        scores = _univar_numeric(stats)
    elif col_type == "categorical":
        scores = _univar_categorical(stats)
    elif col_type == "date":
        return "line"
    else:
        return "table"

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "table"


def _univar_numeric(s: dict) -> dict:
    sc = dict(histogram=0, box=0, bar=0, pie=0, line=0)

    skew      = abs(s.get("skewness") or 0)
    entropy   = s.get("numeric_entropy") or 0
    variance  = s.get("variance_score") or 0
    cv        = s.get("cv_pct") or 0
    outlier_r = s.get("outlier_ratio") or 0
    monotonic = s.get("monotonic", False)
    n_valid   = s.get("n_valid") or 0

    sc["histogram"] += _W["hist_base"]

    if skew > _T["skew_high"]:
        sc["histogram"] += _W["hist_skew"]
        sc["box"]       += _W["box_skew"]

    if entropy > _T["entropy_numeric_high"]:
        sc["histogram"] += _W["hist_entropy"]

    if variance > _T["variance_high"] or cv > _T["cv_high"]:
        sc["histogram"] += _W["hist_variance"]
        sc["box"]       += _W["box_variance"]

    if outlier_r > _T["outlier_pct_high"]:
        sc["box"] += _W["box_outliers"]

    if monotonic:
        sc["line"] += _W["line_monotonic"]

    # many distinct numeric values → histogram beats bar
    if n_valid > 20:
        sc["histogram"] += _W["hist_many_vals"]

    return sc


def _univar_categorical(s: dict) -> dict:
    sc = dict(histogram=0, box=0, bar=0, pie=0, line=0)

    unique     = s.get("unique_count") or 0
    entropy    = s.get("categorical_entropy") or 0
    balance    = s.get("balance_score") or 0
    cardinality= s.get("cardinality_ratio") or 0

    sc["bar"] += _W["bar_base"]

    # pie: few categories
    if unique <= _T["pie_max_unique"]:
        sc["pie"] += _W["pie_small"]

    # binary → pie is very effective
    if unique == 2:
        sc["pie"] += _W["pie_binary"]

    # balanced distribution → pie works well
    if balance > 0.6:
        sc["pie"] += _W["pie_balance"]

    # high cardinality → bar is safer, kill pie
    if cardinality > 0.5 or unique > _T["bar_few_unique"]:
        sc["pie"] += _W["pie_penalty_large"]
        sc["bar"] += _W["bar_few_cats"]

    if entropy > _T["entropy_cat_high"]:
        sc["bar"] += _W["bar_entropy"]

    return sc


# ══════════════════════════════════════════════════════════════
# BIVARIATE
# ══════════════════════════════════════════════════════════════

def choose_bivariate_chart(
    type_x: str,
    type_y: str,
    stats_x: dict | None = None,
    stats_y: dict | None = None,
) -> str:
    sc = dict(scatter=0, line=0, bar=0, stacked_bar=0, box=0)

    sx = stats_x or {}
    sy = stats_y or {}

    # ── numeric × numeric ────────────────────────────────────
    if type_x == "numeric" and type_y == "numeric":
        sc["scatter"] += _W["scatter_base"]

        if _high_entropy(sx) or _high_entropy(sy):
            sc["scatter"] += _W["scatter_entropy"]

        if _is_monotonic(sx) or _is_monotonic(sy):
            sc["line"] += 2

        if _has_outliers(sx) or _has_outliers(sy):
            sc["box"] += 1

    # ── date × numeric ───────────────────────────────────────
    elif (type_x == "date" and type_y == "numeric") or \
         (type_y == "date" and type_x == "numeric"):
        sc["line"] += _W["line_time"]

    # ── categorical × numeric ────────────────────────────────
    elif type_x == "categorical" and type_y == "numeric":
        _add(sc, _cat_num_scores(sx, sy))

    elif type_y == "categorical" and type_x == "numeric":
        _add(sc, _cat_num_scores(sy, sx))

    # ── categorical × categorical ────────────────────────────
    elif type_x == "categorical" and type_y == "categorical":
        sc["stacked_bar"] += _W["stacked_base"]
        # if both have few categories prefer grouped bar
        ux = sx.get("unique_count") or 999
        uy = sy.get("unique_count") or 999
        if ux <= 6 and uy <= 6:
            sc["bar"] += 2

    best = max(sc, key=sc.get)
    return best if sc[best] > 0 else "table"


def _cat_num_scores(cat_s: dict, num_s: dict) -> dict:
    sc = dict(scatter=0, line=0, bar=0, stacked_bar=0, box=0)
    unique = cat_s.get("unique_count") or 0

    if unique <= _T["cat_many_unique"]:
        sc["bar"] += _W["grouped_bar_few"]
    else:
        sc["box"] += _W["grouped_box_many"]

    if (num_s.get("variance_score") or 0) > _T["variance_high"]:
        sc["box"] += _W["grouped_box_var"]

    if _has_outliers(num_s):
        sc["box"] += 1

    return sc


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _add(base: dict, delta: dict):
    for k, v in delta.items():
        base[k] = base.get(k, 0) + v


def _is_monotonic(s: dict) -> bool:
    return bool(s.get("monotonic", False))


def _has_outliers(s: dict) -> bool:
    return (s.get("outlier_ratio") or 0) > _T["outlier_pct_high"]


def _high_entropy(s: dict) -> bool:
    return (s.get("numeric_entropy") or 0) > _T["entropy_numeric_high"]
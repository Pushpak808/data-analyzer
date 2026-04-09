# =========================================================
# CHART SELECTOR — INTELLIGENT SCORING ENGINE
# =========================================================

_WEIGHTS = {
    "histogram_base": 2,
    "skew_bonus": 2,
    "entropy_bonus": 1,
    "variance_bonus": 1,

    "box_outlier": 3,
    "box_skew": 1,

    "line_monotonic": 3,

    "bar_base": 2,
    "bar_entropy": 1,
    "bar_many_categories": 2,

    "pie_small": 2,
    "pie_balance": 1,
    "pie_penalty_high_cardinality": -2,

    "scatter_base": 3,
    "line_time": 4,
    "stacked_base": 3,
}


_THRESHOLDS = {
    "skew_high": 1.0,
    "entropy_numeric": 2.0,
    "entropy_categorical": 1.5,
    "outlier_ratio": 3.0,
    "pie_max_unique": 6,
    "bar_many_unique": 8,
    "cat_many_unique": 10,
    "variance_high": 1.0,
    "balance_pie": 0.5,
    "cardinality_high": 0.5,
}


# =========================================================
# UNIVARIATE
# =========================================================

def choose_univariate_chart(stats, col_type):
    scores = {
        "histogram": 0,
        "box": 0,
        "bar": 0,
        "pie": 0,
        "line": 0,
    }

    if col_type == "numeric":
        scores = _score_numeric_univariate(stats)

    elif col_type == "categorical":
        scores = _score_categorical_univariate(stats)

    elif col_type == "date":
        scores["line"] = _WEIGHTS["line_time"]

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "table"


# =========================================================
# BIVARIATE
# =========================================================

def choose_bivariate_chart(type_x, type_y, stats_x=None, stats_y=None):
    scores = {
        "scatter": 0,
        "line": 0,
        "bar": 0,
        "stacked_bar": 0,
        "box": 0,
    }

    if type_x == "numeric" and type_y == "numeric":
        scores["scatter"] += _WEIGHTS["scatter_base"]

        if _is_monotonic(stats_x) or _is_monotonic(stats_y):
            scores["line"] += 2

        if _has_outliers(stats_x) or _has_outliers(stats_y):
            scores["box"] += 1

    elif type_x == "categorical" and type_y == "numeric":
        delta = _categorical_numeric_scores(stats_x, stats_y)
        _accumulate(scores, delta)

    elif type_y == "categorical" and type_x == "numeric":
        delta = _categorical_numeric_scores(stats_y, stats_x)
        _accumulate(scores, delta)

    elif type_x == "date" and type_y == "numeric":
        scores["line"] += _WEIGHTS["line_time"]

    elif type_y == "date" and type_x == "numeric":
        scores["line"] += _WEIGHTS["line_time"]

    elif type_x == "categorical" and type_y == "categorical":
        scores["stacked_bar"] += _WEIGHTS["stacked_base"]

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "table"


# =========================================================
# NUMERIC UNIVARIATE
# =========================================================

def _score_numeric_univariate(stats):
    scores = {
        "histogram": 0,
        "box": 0,
        "bar": 0,
        "pie": 0,
        "line": 0,
    }

    skew = abs(stats.get("skewness", 0))
    entropy = stats.get("numeric_entropy", 0)
    variance = stats.get("variance_score", 0)
    outliers = stats.get("outlier_ratio", 0)
    monotonic = stats.get("monotonic", False)

    scores["histogram"] += _WEIGHTS["histogram_base"]

    if skew > _THRESHOLDS["skew_high"]:
        scores["histogram"] += _WEIGHTS["skew_bonus"]
        scores["box"] += _WEIGHTS["box_skew"]

    if entropy > _THRESHOLDS["entropy_numeric"]:
        scores["histogram"] += _WEIGHTS["entropy_bonus"]

    if variance > _THRESHOLDS["variance_high"]:
        scores["histogram"] += _WEIGHTS["variance_bonus"]

    if outliers > _THRESHOLDS["outlier_ratio"]:
        scores["box"] += _WEIGHTS["box_outlier"]

    if monotonic:
        scores["line"] += _WEIGHTS["line_monotonic"]

    return scores


# =========================================================
# CATEGORICAL UNIVARIATE
# =========================================================

def _score_categorical_univariate(stats):
    scores = {
        "histogram": 0,
        "box": 0,
        "bar": 0,
        "pie": 0,
        "line": 0,
    }

    unique = stats.get("unique_count", 0)
    balance = stats.get("balance_score", 0)
    entropy = stats.get("categorical_entropy", 0)
    cardinality = stats.get("cardinality_ratio", 0)

    scores["bar"] += _WEIGHTS["bar_base"]

    if unique <= _THRESHOLDS["pie_max_unique"]:
        scores["pie"] += _WEIGHTS["pie_small"]

    if balance > _THRESHOLDS["balance_pie"]:
        scores["pie"] += _WEIGHTS["pie_balance"]

    if unique > _THRESHOLDS["bar_many_unique"]:
        scores["bar"] += _WEIGHTS["bar_many_categories"]

    if entropy > _THRESHOLDS["entropy_categorical"]:
        scores["bar"] += _WEIGHTS["bar_entropy"]

    if cardinality > _THRESHOLDS["cardinality_high"]:
        scores["pie"] += _WEIGHTS["pie_penalty_high_cardinality"]

    return scores


# =========================================================
# CATEGORICAL × NUMERIC
# =========================================================

def _categorical_numeric_scores(cat_stats, num_stats):
    scores = {
        "scatter": 0,
        "line": 0,
        "bar": 0,
        "stacked_bar": 0,
        "box": 0,
    }

    unique = cat_stats.get("unique_count", 0)

    if unique <= _THRESHOLDS["cat_many_unique"]:
        scores["bar"] += 3

    if unique > _THRESHOLDS["cat_many_unique"]:
        scores["box"] += 3

    if num_stats and num_stats.get("variance_score", 0) > _THRESHOLDS["variance_high"]:
        scores["box"] += 1

    if _has_outliers(num_stats):
        scores["box"] += 1

    return scores


# =========================================================
# HELPERS
# =========================================================

def _accumulate(base, delta):
    for k, v in delta.items():
        base[k] += v


def _is_monotonic(stats):
    if not stats:
        return False
    return stats.get("monotonic", False)


def _has_outliers(stats):
    if not stats:
        return False
    return stats.get("outlier_ratio", 0) > _THRESHOLDS["outlier_ratio"]
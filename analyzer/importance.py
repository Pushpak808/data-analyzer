# =========================================================
# IMPORTANCE SCORER (uses stats only)
# =========================================================

_SCORE_MIN = -5
_SCORE_MAX = 7


def score_columns(stats):
    """
    stats: output of compute_stats()
    returns column importance ranking
    """

    results = {}

    for col, col_stats in stats.items():
        col_type = col_stats["type"]

        score, reasons = _score_column(col_stats, col_type)

        # clamp score to stable range
        score = max(_SCORE_MIN, min(score, _SCORE_MAX))

        label = _label(score)

        results[col] = {
            "type": col_type,
            "score": round(score, 2),
            "label": label,
            "missing_rate": col_stats.get("missing_rate", 0),
            "reasons": reasons,
        }

    return results


# =========================================================
# SCORE ENGINE
# =========================================================

def _score_column(stats, col_type):
    score = 0
    reasons = []

    missing = stats.get("missing_rate", 0)

    # -----------------------------------------------------
    # MISSING
    # -----------------------------------------------------
    if missing > 70:
        score -= 3
        reasons.append("Very high missing values")

    elif missing > 40:
        score -= 1
        reasons.append("High missing values")

    # -----------------------------------------------------
    # NUMERIC
    # -----------------------------------------------------
    if col_type == "numeric":
        score += 1

        variance = stats.get("variance_score", 0)
        skew = abs(stats.get("skewness", 0))
        outliers = stats.get("outlier_ratio", 0)

        if variance > 1:
            score += 2
            reasons.append("High variance")

        elif variance < 0.01:
            score -= 2
            reasons.append("Near constant values")

        if skew > 1:
            score += 1
            reasons.append("Skewed distribution")

        if outliers > 5:
            score -= 1
            reasons.append("Many outliers")

    # -----------------------------------------------------
    # CATEGORICAL
    # -----------------------------------------------------
    elif col_type == "categorical":
        score += 1

        entropy = stats.get("categorical_entropy", 0)
        unique = stats.get("unique_count", 0)
        balance = stats.get("balance_score", 0)
        cardinality = stats.get("cardinality_ratio", 0)

        if entropy > 1.5:
            score += 2
            reasons.append("Good category diversity")

        if balance > 0.5:
            score += 1
            reasons.append("Balanced categories")

        if unique > 50 or cardinality > 0.8:
            score -= 2
            reasons.append("Too many categories")

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------
    elif col_type == "date":
        score += 2
        reasons.append("Time series data")

        if stats.get("is_sorted"):
            score += 1
            reasons.append("Chronological")

    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------
    elif col_type == "text":
        score -= 1
        reasons.append("Free text column")

    return score, reasons


# =========================================================
# LABEL
# =========================================================

def _label(score):
    if score >= 3:
        return "important"

    if score >= 1:
        return "moderate"

    return "low"
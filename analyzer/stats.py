import pandas as pd
import numpy as np

def detect_type(series):
    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # try convert to datetime
    try:
        parsed = pd.to_datetime(series.dropna().head(20), errors="raise")
        return "date"
    except:
        pass

    # categorical vs text
    unique = series.nunique(dropna=True)
    total = len(series)

    if total == 0:
        return "text"

    ratio = unique / total

    if ratio < 0.5 and unique < 50:
        return "categorical"

    return "text"



def compute_stats(df: pd.DataFrame) -> dict:
    return {col: _col_stats(df[col]) for col in df.columns}


def _col_stats(series: pd.Series) -> dict:
    col_type = detect_type(series)

    n = len(series)
    missing = int(series.isna().sum())

    base = {
        "type": col_type,
        "total": n,
        "missing": missing,
        "missing_rate": round(missing / n * 100, 1) if n else 0.0,
    }

    dispatch = {
        "numeric": _numeric_stats,
        "categorical": _categorical_stats,
        "date": _date_stats,
    }

    handler = dispatch.get(col_type, _text_stats)
    return handler(series, base)


# =========================================================
# NUMERIC
# =========================================================

def _numeric_stats(series: pd.Series, base: dict) -> dict:
    clean = _get_clean(series)
    if clean is None:
        return base

    n = len(clean)

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    # outliers
    outliers_mask = (clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))
    outlier_count = int(outliers_mask.sum())
    outlier_ratio = round(outlier_count / n * 100, 2)

    # skew
    skew = clean.skew()
    skew_type = _skew_type(skew)

    # distribution type
    abs_skew = abs(skew)
    if abs_skew < 0.5:
        distribution = "normal_like"
    elif abs_skew < 1:
        distribution = "mild_skew"
    else:
        distribution = "highly_skewed"

    # monotonic trend
    monotonic = (
        clean.is_monotonic_increasing
        or clean.is_monotonic_decreasing
    )

    # variance score (correct CV²)
    variance = clean.var()
    mean = clean.mean()
    variance_score = float(variance / mean ** 2) if mean != 0 else float("inf")

    # entropy (numeric)
    numeric_entropy = _numeric_entropy(clean)

    return {
        **base,
        "mean": _safe_round(mean),
        "median": _safe_round(clean.median()),
        "std": _safe_round(clean.std()),
        "variance": _safe_round(variance),
        "min": _safe_round(clean.min()),
        "max": _safe_round(clean.max()),
        "range": _safe_round(clean.max() - clean.min()),
        "q1": _safe_round(q1),
        "q3": _safe_round(q3),
        "iqr": _safe_round(iqr),
        "skewness": _safe_round(skew),
        "skew_type": skew_type,
        "distribution": distribution,
        "outliers": outlier_count,
        "outlier_ratio": outlier_ratio,
        "variance_score": _safe_round(variance_score),
        "monotonic": monotonic,
        "numeric_entropy": _safe_round(numeric_entropy),
    }


# =========================================================
# CATEGORICAL
# =========================================================

def _categorical_stats(series: pd.Series, base: dict) -> dict:
    clean = _get_clean(series, cast=str)
    if clean is None:
        return base

    n = len(clean)

    value_counts = clean.value_counts()
    probs = value_counts.values / n

    categorical_entropy = -np.sum(probs * np.log2(probs))
    dominant = probs[0]

    balance_score = 1 - dominant
    cardinality_ratio = len(value_counts) / n

    return {
        **base,
        "unique_count": int(len(value_counts)),
        "cardinality_ratio": _safe_round(cardinality_ratio),
        "dominant_value": value_counts.index[0],
        "dominant_rate": round(dominant * 100, 2),
        "categorical_entropy": _safe_round(categorical_entropy),
        "balance_score": _safe_round(balance_score),
        "top_values": value_counts.head(10).index.tolist(),
        "top_counts": value_counts.head(10).values.tolist(),
    }


# =========================================================
# DATE
# =========================================================

def _date_stats(series: pd.Series, base: dict) -> dict:
    parsed = pd.to_datetime(series, errors="coerce")
    clean = _get_clean(parsed)

    if clean is None:
        return base

    return {
        **base,
        "min_date": str(clean.min().date()),
        "max_date": str(clean.max().date()),
        "range_days": int((clean.max() - clean.min()).days),
        "unique_dates": int(clean.nunique()),
        "is_sorted": clean.is_monotonic_increasing,
    }


# =========================================================
# TEXT
# =========================================================

def _text_stats(series: pd.Series, base: dict) -> dict:
    clean = _get_clean(series, cast=str)

    if clean is None:
        return base

    return {
        **base,
        "unique_count": int(clean.nunique()),
        "avg_length": _safe_round(clean.str.len().mean()),
        "max_length": int(clean.str.len().max()),
        "min_length": int(clean.str.len().min()),
    }


# =========================================================
# HELPERS
# =========================================================

def _get_clean(series, cast=None):
    clean = series.dropna()
    if cast:
        clean = clean.astype(cast)
    return clean if len(clean) > 0 else None


def _numeric_entropy(series):
    hist, _ = np.histogram(series, bins=20)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))


def _skew_type(skew):
    if skew > 1:
        return "right_skewed"
    if skew < -1:
        return "left_skewed"
    return "symmetric"


def _safe_round(val, decimals=4):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), decimals)
    except Exception:
        return None
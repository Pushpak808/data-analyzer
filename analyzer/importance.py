import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy


def score_columns(df: pd.DataFrame) -> dict:
    results = {}

    for col in df.columns:
        series = df[col]
        score = 0
        reasons = []
        col_type = detect_type(series)

        missing_rate = series.isna().mean()

        # ── MISSING VALUES ──
        if missing_rate > 0.7:
            score -= 3
            reasons.append(f"High missing rate ({missing_rate:.0%})")
        elif missing_rate > 0.4:
            score -= 1
            reasons.append(f"Moderate missing rate ({missing_rate:.0%})")

        # ── NUMERIC CHECKS ──
        if col_type == "numeric":
            clean = series.dropna()
            std = clean.std()
            mean = clean.mean()
            unique_ratio = clean.nunique() / len(clean)

            if std == 0:
                score -= 3
                reasons.append("Constant column — zero variance")

            elif mean != 0:
                cv = std / abs(mean)
                if cv < 0.01:
                    score -= 2
                    reasons.append("Near-zero variance")
                elif cv > 0.5:
                    score += 1
                    reasons.append("High variability — informative")

            if unique_ratio > 0.95:
                score -= 2
                reasons.append("Likely an ID column")
            elif unique_ratio < 0.05:
                score += 1
                reasons.append("Low cardinality — good for grouping")

            score += 1  # numeric columns are generally useful

        # ── CATEGORICAL CHECKS ──
        elif col_type == "categorical":
            clean = series.dropna().astype(str)
            unique_ratio = clean.nunique() / len(clean)
            value_counts = clean.value_counts(normalize=True)
            dominant = value_counts.iloc[0] if len(value_counts) else 1.0

            if unique_ratio > 0.95:
                score -= 2
                reasons.append("Likely an ID or free-text column")

            if dominant > 0.99:
                score -= 2
                reasons.append("Almost uniform — one value dominates")

            probs = value_counts.values
            ent = scipy_entropy(probs, base=2)
            if ent < 0.3:
                score -= 1
                reasons.append(f"Very low entropy ({ent:.2f}) — little variation")
            elif ent > 1.5:
                score += 1
                reasons.append(f"Good entropy ({ent:.2f}) — diverse values")

            score += 1  # categorical columns useful for segmentation

        # ── DATE CHECKS ──
        elif col_type == "date":
            score += 2
            reasons.append("Date column — useful for time-series analysis")

        # ── TEXT CHECKS ──
        elif col_type == "text":
            score -= 1
            reasons.append("Free text — low analytical value")

        # ── FINAL LABEL ──
        if score >= 2:
            label = "important"
        elif score >= 0:
            label = "moderate"
        else:
            label = "low"

        results[col] = {
            "type": col_type,
            "score": score,
            "label": label,
            "missing_rate": round(missing_rate * 100, 1),
            "reasons": reasons,
        }

    return results


def detect_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # try parsing as date
    sample = series.dropna().astype(str).head(50)
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True)
        if parsed.notna().mean() > 0.8:
            return "date"
    except Exception:
        pass

    # check cardinality for categorical vs text
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return "text"

    unique_ratio = clean.nunique() / len(clean)
    avg_length = clean.str.len().mean()

    if avg_length > 40:
        return "text"

    if unique_ratio < 0.5:
        return "categorical"

    return "text"
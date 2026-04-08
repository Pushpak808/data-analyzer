import pandas as pd
import numpy as np
from analyzer.importance import detect_type


def compute_stats(df: pd.DataFrame) -> dict:
    results = {}

    for col in df.columns:
        series = df[col]
        col_type = detect_type(series)
        missing = int(series.isna().sum())
        missing_rate = round(missing / len(series) * 100, 1)

        base = {
            "type": col_type,
            "total": len(series),
            "missing": missing,
            "missing_rate": missing_rate,
        }

        if col_type == "numeric":
            clean = series.dropna()
            q1 = float(clean.quantile(0.25))
            q3 = float(clean.quantile(0.75))
            iqr = q3 - q1

            # outlier count using IQR method
            outliers = int(((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).sum())

            results[col] = {
                **base,
                "mean":     safe_round(clean.mean()),
                "median":   safe_round(clean.median()),
                "std":      safe_round(clean.std()),
                "variance": safe_round(clean.var()),
                "min":      safe_round(clean.min()),
                "max":      safe_round(clean.max()),
                "q1":       safe_round(q1),
                "q3":       safe_round(q3),
                "iqr":      safe_round(iqr),
                "skewness": safe_round(clean.skew()),
                "kurtosis": safe_round(clean.kurt()),
                "outliers": outliers,
                "sum":      safe_round(clean.sum()),
            }

        elif col_type == "categorical":
            clean = series.dropna().astype(str)
            value_counts = clean.value_counts()
            top10 = value_counts.head(10)

            results[col] = {
                **base,
                "unique_count":    int(clean.nunique()),
                "top_values":      top10.index.tolist(),
                "top_counts":      top10.values.tolist(),
                "top_frequencies": [round(v / len(clean) * 100, 1) for v in top10.values],
                "dominant_value":  str(value_counts.index[0]) if len(value_counts) else None,
                "dominant_rate":   round(value_counts.iloc[0] / len(clean) * 100, 1) if len(value_counts) else None,
            }

        elif col_type == "date":
            parsed = pd.to_datetime(series, errors="coerce")
            clean = parsed.dropna()

            results[col] = {
                **base,
                "min_date":   str(clean.min().date()) if len(clean) else None,
                "max_date":   str(clean.max().date()) if len(clean) else None,
                "range_days": int((clean.max() - clean.min()).days) if len(clean) else None,
                "unique_dates": int(clean.nunique()),
            }

        else:
            # text column
            clean = series.dropna().astype(str)
            results[col] = {
                **base,
                "unique_count": int(clean.nunique()),
                "avg_length":   safe_round(clean.str.len().mean()),
                "max_length":   int(clean.str.len().max()) if len(clean) else None,
                "min_length":   int(clean.str.len().min()) if len(clean) else None,
            }

    return results


def safe_round(val, decimals=4):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), decimals)
    except Exception:
        return None
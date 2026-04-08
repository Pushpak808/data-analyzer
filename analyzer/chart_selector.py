import pandas as pd
import numpy as np


def choose_univariate_chart(series: pd.Series, col_type: str):
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return None

    unique = clean.nunique()
    unique_ratio = unique / n

    scores = {
        "histogram": 0,
        "bar": 0,
        "pie": 0,
        "box": 0,
    }

    # categorical-like
    if unique <= 8:
        scores["pie"] += 3
        scores["bar"] += 2

    if unique <= 20:
        scores["bar"] += 2

    # numeric continuous
    if col_type == "numeric":
        scores["histogram"] += 2
        scores["box"] += 1

        if unique_ratio > 0.5:
            scores["histogram"] += 2

        # skewness bonus
        skew = clean.skew()
        if abs(skew) > 1:
            scores["histogram"] += 1

    # categorical
    if col_type == "categorical":
        scores["bar"] += 2
        if unique <= 6:
            scores["pie"] += 2

    # choose best
    best = max(scores, key=scores.get)
    return best


def choose_bivariate_chart(type_x, type_y):
    # numeric vs numeric
    if type_x == "numeric" and type_y == "numeric":
        return "scatter"

    # categorical vs numeric
    if type_x == "categorical" and type_y == "numeric":
        return "bar"

    if type_y == "categorical" and type_x == "numeric":
        return "bar"

    # date vs numeric
    if type_x == "date" and type_y == "numeric":
        return "line"

    if type_y == "date" and type_x == "numeric":
        return "line"

    # categorical vs categorical
    if type_x == "categorical" and type_y == "categorical":
        return "stacked_bar"

    return None
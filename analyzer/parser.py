import pandas as pd
import json
import io


def parse_file(contents: bytes, ext: str) -> pd.DataFrame:
    if ext == "csv":
        df = pd.read_csv(io.BytesIO(contents))

    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(contents))

    elif ext == "json":
        data = json.loads(contents.decode("utf-8"))
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("JSON must be an array or object")

    df.columns = df.columns.str.strip()

    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df
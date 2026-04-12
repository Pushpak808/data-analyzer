import numpy as np

def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_native(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_native(v) for v in obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    return obj
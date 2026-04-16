from typing import Dict, List

import numpy as np


def summary_stats(ts: List[float]) -> Dict[str, float]:
    arr = np.array(ts, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def autocorr(ts: List[float], lag: int = 1) -> float:
    if lag <= 0 or lag >= len(ts):
        return 0.0
    x = np.array(ts[:-lag], dtype=float)
    y = np.array(ts[lag:], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def volatility(ts: List[float]) -> float:
    arr = np.array(ts, dtype=float)
    if len(arr) < 2:
        return 0.0
    diff = np.diff(arr)
    return float(np.std(diff))

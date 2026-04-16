from typing import Dict, List

import numpy as np


def corr_relation(ts1: List[float], ts2: List[float], lag: int = 0) -> Dict[str, float]:
    a = np.array(ts1, dtype=float)
    b = np.array(ts2, dtype=float)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if lag > 0 and lag < n:
        a = a[:-lag]
        b = b[lag:]
    elif lag < 0 and -lag < n:
        a = a[-lag:]
        b = b[: lag if lag != 0 else None]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return {"corr": 0.0}
    return {"corr": float(np.corrcoef(a, b)[0, 1])}


def cross_correlation(ts1: List[float], ts2: List[float], max_lag: int = 24) -> Dict[str, float]:
    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        c = abs(corr_relation(ts1, ts2, lag)["corr"])
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return {"best_lag": float(best_lag), "best_corr": float(best_corr)}


def granger_like(ts1: List[float], ts2: List[float], max_lag: int = 4) -> Dict[str, object]:
    # Lightweight proxy for causality-style questions.
    best = cross_correlation(ts1, ts2, max_lag=max_lag)
    lag = int(best["best_lag"])
    if lag > 0:
        relation = "ts1_leads_ts2"
    elif lag < 0:
        relation = "ts2_leads_ts1"
    else:
        relation = "synchronous_or_unclear"
    return {"relation": relation, "lag": lag, "score": float(best["best_corr"])}

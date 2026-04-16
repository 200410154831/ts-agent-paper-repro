from typing import Dict, List

import numpy as np

from tools.numeric import autocorr


def trend_classifier(ts: List[float]) -> Dict[str, float]:
    x = np.arange(len(ts), dtype=float)
    y = np.array(ts, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(ts) >= 2 else 0.0
    if slope > 0.01:
        label = "up"
    elif slope < -0.01:
        label = "down"
    else:
        label = "flat"
    return {"label": label, "slope": slope}


def anomaly_classifier(ts: List[float]) -> Dict[str, object]:
    arr = np.array(ts, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0:
        return {"has_anomaly": False, "indices": [], "segment": "middle"}
    z = np.abs((arr - mean) / std)
    idxs = np.where(z > 2.8)[0].tolist()
    if not idxs:
        return {"has_anomaly": False, "indices": [], "segment": "middle"}
    pos = float(np.mean(idxs)) / max(1, len(arr) - 1)
    if pos < 1 / 3:
        seg = "beginning"
    elif pos < 2 / 3:
        seg = "middle"
    else:
        seg = "end"
    return {"has_anomaly": True, "indices": idxs, "segment": seg}


def seasonality_detector(ts: List[float], max_lag: int = 24) -> Dict[str, object]:
    best_lag = 1
    best = -1.0
    max_lag = min(max_lag, max(2, len(ts) // 2))
    for lag in range(2, max_lag + 1):
        val = autocorr(ts, lag)
        if val > best:
            best = val
            best_lag = lag
    return {"period": best_lag, "strength": float(best), "has_seasonality": bool(best > 0.4)}


def stationarity_test(ts: List[float]) -> Dict[str, object]:
    arr = np.array(ts, dtype=float)
    if len(arr) < 8:
        return {"stationary": False, "score": 0.0}
    first = arr[: len(arr) // 2]
    second = arr[len(arr) // 2 :]
    mean_shift = abs(np.mean(first) - np.mean(second))
    std_scale = np.std(arr) + 1e-8
    score = float(mean_shift / std_scale)
    return {"stationary": score < 0.5, "score": score}


def noise_profile(ts: List[float]) -> Dict[str, object]:
    ac1 = autocorr(ts, 1)
    if abs(ac1) < 0.15:
        label = "white"
    elif ac1 > 0.4:
        label = "red"
    else:
        label = "other"
    return {"label": label, "lag1_autocorr": ac1}

from typing import List


def slice_series(ts: List[float], start: int, end: int) -> List[float]:
    start = max(0, int(start))
    end = min(len(ts), int(end))
    if start >= end:
        return []
    return ts[start:end]


def segment_series(ts: List[float], k: int) -> List[List[float]]:
    k = max(1, int(k))
    n = len(ts)
    base = n // k
    rem = n % k
    out: List[List[float]] = []
    idx = 0
    for i in range(k):
        length = base + (1 if i < rem else 0)
        out.append(ts[idx : idx + length])
        idx += length
    return out

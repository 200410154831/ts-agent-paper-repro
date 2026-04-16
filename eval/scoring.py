import json
from collections import defaultdict
from typing import Dict, List


TABLE1_ORDER = [
    "Pattern Recognition",
    "Noise Understanding",
    "Anomaly Detection",
    "Similarity Analysis",
    "Causality Analysis",
]


def load_jsonl(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def category_accuracy(rows: List[Dict[str, object]]) -> Dict[str, float]:
    hit = defaultdict(int)
    total = defaultdict(int)
    for r in rows:
        c = str(r["category"])
        total[c] += 1
        if bool(r["is_correct"]):
            hit[c] += 1
    out: Dict[str, float] = {}
    for c in TABLE1_ORDER:
        out[c] = (hit[c] / total[c]) if total[c] else 0.0
    out["Overall"] = (sum(hit.values()) / sum(total.values())) if total else 0.0
    out["Total"] = float(sum(total.values()))
    return out

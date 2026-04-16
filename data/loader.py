import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


CATEGORY_MAP = {
    "Pattern Recognition": "Pattern Recognition",
    "Noise Understanding": "Noise Understanding",
    "Anolmaly Detection": "Anomaly Detection",
    "Anomaly Detection": "Anomaly Detection",
    "Similarity Analysis": "Similarity Analysis",
    "Causality Analysis": "Causality Analysis",
}


@dataclass
class TimeSeriesExamItem:
    id: int
    question: str
    options: List[str]
    answer: str
    category: str
    subcategory: str
    tid: Optional[int]
    ts: Optional[List[float]]
    ts1: Optional[List[float]]
    ts2: Optional[List[float]]
    raw: Dict[str, Any]


def _to_float_list(values: Any) -> Optional[List[float]]:
    if values is None:
        return None
    if not isinstance(values, list):
        return None
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


def normalize_category(category: str) -> str:
    return CATEGORY_MAP.get(category, category)


def load_timeseries_exam(path: str) -> List[TimeSeriesExamItem]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    items: List[TimeSeriesExamItem] = []
    for row in rows:
        items.append(
            TimeSeriesExamItem(
                id=int(row["id"]),
                question=row["question"],
                options=list(row["options"]),
                answer=str(row["answer"]),
                category=normalize_category(str(row["category"])),
                subcategory=str(row.get("subcategory", "")),
                tid=row.get("tid"),
                ts=_to_float_list(row.get("ts")),
                ts1=_to_float_list(row.get("ts1")),
                ts2=_to_float_list(row.get("ts2")),
                raw=row,
            )
        )
    return items


def series_payload(item: TimeSeriesExamItem) -> Dict[str, Any]:
    if item.ts is not None:
        return {"ts": item.ts}
    return {"ts1": item.ts1, "ts2": item.ts2}

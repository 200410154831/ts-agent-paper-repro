import json
from dataclasses import dataclass
from typing import Dict, List

from data.loader import TimeSeriesExamItem
from data.loader import series_payload
from llm.deepseek_client import DeepSeekClient
from agent.prompt_meta import build_non_answer_fields


@dataclass
class Intent:
    task: str
    schema: str
    required_predicates: List[str]


def _rule_based_intent(item: TimeSeriesExamItem) -> Intent:
    q = item.question.lower()
    sub = item.subcategory.lower()
    cat = item.category.lower()

    if "granger" in sub or "causality" in cat:
        return Intent("causality", "single_choice", ["lead_relation"])
    if "similarity" in cat or "distributional" in sub or "shape" in sub:
        return Intent("similarity", "single_choice", ["corr_or_distance"])
    if "anomaly" in cat or "anomaly" in sub:
        return Intent("anomaly", "single_choice", ["has_anomaly", "anomaly_segment_or_type"])
    if "noise" in cat or "white noise" in q or "red noise" in q:
        return Intent("noise", "single_choice", ["noise_type"])
    if "trend" in q or "trend" in sub:
        return Intent("trend", "single_choice", ["trend_direction"])
    if "stationarity" in q or "stationarity" in sub:
        return Intent("stationarity", "single_choice", ["stationary_flag"])
    if "cycle" in q or "season" in q:
        return Intent("seasonality", "single_choice", ["seasonality"])
    return Intent("generic", "single_choice", ["basic_stats"])


def detect_intent(item: TimeSeriesExamItem, llm: DeepSeekClient) -> Intent:
    meta = build_non_answer_fields(item)
    prompt = (
        "You are an intent extractor for a time-series agent.\n"
        "Return valid json object with keys: task, schema, required_predicates.\n"
        "task should be one of: trend, seasonality, anomaly, noise, stationarity, similarity, causality, generic.\n"
        "schema should be one of: single_choice, multi_choice, free_text.\n"
        "required_predicates must be a JSON array of strings.\n"
        "JSON example: {\"task\":\"anomaly\",\"schema\":\"single_choice\",\"required_predicates\":[\"has_anomaly\",\"anomaly_segment_or_type\"]}\n"
        f"Question: {item.question}\n"
        f"Non-answer fields: {json.dumps(meta, ensure_ascii=False)}\n"
        f"Time series data: {json.dumps(series_payload(item), ensure_ascii=False)}\n"
    )
    try:
        raw = llm.generate(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        obj = json.loads(raw)
        task = str(obj.get("task", "")).strip().lower()
        schema = str(obj.get("schema", "")).strip()
        preds = obj.get("required_predicates", [])
        if task not in {"trend", "seasonality", "anomaly", "noise", "stationarity", "similarity", "causality", "generic"}:
            raise ValueError("invalid task")
        if schema not in {"single_choice", "multi_choice", "free_text"}:
            schema = "single_choice"
        if not isinstance(preds, list) or not preds:
            raise ValueError("invalid predicates")
        preds = [str(x) for x in preds if str(x).strip()]
        if not preds:
            raise ValueError("empty predicates")
        return Intent(task, schema, preds)
    except Exception:
        return _rule_based_intent(item)


def update_covered_predicates(evidence: Dict[str, object], required: List[str]) -> List[str]:
    covered: List[str] = []
    for r in required:
        if r in evidence and evidence[r] is not None:
            covered.append(r)
    return covered

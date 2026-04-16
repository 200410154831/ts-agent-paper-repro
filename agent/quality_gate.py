import json
from typing import Any, Dict, List, Optional, Tuple

from llm.deepseek_client import DeepSeekClient


def _rule_based_check_final_answer(
    pred: str,
    options: List[str],
    gap: List[str],
    evidence: Dict[str, Any],
) -> Tuple[bool, List[str], Optional[int]]:
    reasons: List[str] = []
    if pred not in options:
        reasons.append("schema_mismatch")
    if gap:
        reasons.append("missing_predicates:" + ",".join(gap))
    if evidence.get("contradiction", False):
        reasons.append("contradictory_evidence")
    return (len(reasons) == 0, reasons, None)


def check_final_answer(
    llm: DeepSeekClient,
    question: str,
    non_answer_fields: Dict[str, Any],
    pred: str,
    options: List[str],
    gap: List[str],
    evidence: Dict[str, Any],
) -> Tuple[bool, List[str], Optional[int]]:
    prompt = (
        "You are a final quality gate for a time-series reasoning system.\n"
        "Return valid json object with keys: accept, reasons, backtrack_step.\n"
        "accept must be true or false. reasons must be an array of strings.\n"
        "backtrack_step should be an integer step index (1-based) when reject and backtracking is needed; otherwise null.\n"
        "Reject if answer is not in options, or gap predicates are not empty, or evidence is contradictory.\n"
        "JSON example: {\"accept\": false, \"reasons\": [\"missing_predicates:trend_direction\"], \"backtrack_step\": 2}\n"
        f"Question: {question}\n"
        f"Non-answer fields: {json.dumps(non_answer_fields, ensure_ascii=False)}\n"
        f"Predicted answer: {pred}\n"
        f"Options: {json.dumps(options, ensure_ascii=False)}\n"
        f"Gap predicates: {json.dumps(gap, ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
    )
    try:
        raw = llm.generate(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        obj = json.loads(raw)
        accept = bool(obj.get("accept", False))
        reasons = obj.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        backtrack_step = obj.get("backtrack_step", None)
        if backtrack_step is not None:
            try:
                backtrack_step = int(backtrack_step)
            except Exception:
                backtrack_step = None
        return accept, [str(x) for x in reasons], backtrack_step
    except Exception:
        return _rule_based_check_final_answer(pred, options, gap, evidence)

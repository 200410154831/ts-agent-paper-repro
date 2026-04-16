import json
from typing import Any, Dict, Tuple

from llm.deepseek_client import DeepSeekClient


def _fallback_exact_match(gold: str, pred: str) -> bool:
    return str(gold).strip() == str(pred).strip()


def llm_judge_correctness(
    llm: DeepSeekClient,
    question: str,
    non_answer_fields: Dict[str, Any],
    options: list[str],
    gold_answer: str,
    pred_answer: str,
) -> Tuple[bool, str]:
    prompt = (
        "You are an answer judge for multiple-choice time-series questions.\n"
        "Return valid json object with keys: is_correct, reason.\n"
        "is_correct must be true or false.\n"
        "Judge correctness based on whether predicted answer semantically matches the gold answer within provided options.\n"
        "JSON example: {\"is_correct\": true, \"reason\": \"pred matches gold option text\"}\n"
        f"Question: {question}\n"
        f"Non-answer fields: {json.dumps(non_answer_fields, ensure_ascii=False)}\n"
        f"Options: {json.dumps(options, ensure_ascii=False)}\n"
        f"Gold answer: {gold_answer}\n"
        f"Predicted answer: {pred_answer}\n"
    )
    try:
        raw = llm.generate(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        obj = json.loads(raw)
        return bool(obj.get("is_correct", False)), str(obj.get("reason", ""))
    except Exception:
        return _fallback_exact_match(gold_answer, pred_answer), "fallback_exact_match"

import json
from typing import Any, Dict, List

from llm.deepseek_client import DeepSeekClient


def _rule_based_review_step(
    action: str,
    observation: Dict[str, Any],
    gap: List[str],
) -> Dict[str, Any]:
    issues: List[str] = []
    suggestions: List[str] = []

    if "error" in observation:
        issues.append("tool_error")
        suggestions.append("try_alternative_tool_or_fix_input")

    if not observation:
        issues.append("empty_observation")

    if gap:
        suggestions.append("collect_missing_predicates")

    return {
        "tool_suitability": "ok" if not issues else "needs_correction",
        "output_plausibility": "ok" if "error" not in observation else "questionable",
        "evidence_sufficiency": "enough" if not gap else "insufficient",
        "issues": issues,
        "suggestions": suggestions,
    }


def review_step(
    llm: DeepSeekClient,
    question: str,
    non_answer_fields: Dict[str, Any],
    action: str,
    action_input: Dict[str, Any],
    observation: Dict[str, Any],
    evidence: Dict[str, Any],
    gap: List[str],
) -> Dict[str, Any]:
    prompt = (
        "You are a critic for a time-series reasoning agent.\n"
        "Return valid json object with keys: tool_suitability, output_plausibility, evidence_sufficiency, issues, suggestions.\n"
        "tool_suitability should be one of: ok, needs_correction.\n"
        "output_plausibility should be one of: ok, questionable.\n"
        "evidence_sufficiency should be one of: enough, insufficient.\n"
        "issues and suggestions must be arrays of strings.\n"
        "JSON example: {\"tool_suitability\":\"ok\",\"output_plausibility\":\"ok\",\"evidence_sufficiency\":\"insufficient\",\"issues\":[],\"suggestions\":[\"collect_missing_predicates\"]}\n"
        f"Question: {question}\n"
        f"Non-answer fields: {json.dumps(non_answer_fields, ensure_ascii=False)}\n"
        f"Action: {action}\n"
        f"Action input: {json.dumps(action_input, ensure_ascii=False)}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=False)}\n"
        f"Current evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
        f"Gap predicates: {json.dumps(gap, ensure_ascii=False)}\n"
    )
    try:
        raw = llm.generate(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        obj = json.loads(raw)
        for key in ("tool_suitability", "output_plausibility", "evidence_sufficiency"):
            if key not in obj:
                raise ValueError("missing critic key")
        obj["issues"] = obj.get("issues", [])
        obj["suggestions"] = obj.get("suggestions", [])
        if not isinstance(obj["issues"], list):
            obj["issues"] = [str(obj["issues"])]
        if not isinstance(obj["suggestions"], list):
            obj["suggestions"] = [str(obj["suggestions"])]
        return obj
    except Exception:
        return _rule_based_review_step(action, observation, gap)

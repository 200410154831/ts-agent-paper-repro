import json
import re
from typing import Dict, List, Optional

from data.loader import TimeSeriesExamItem
from llm.deepseek_client import DeepSeekClient
from agent.prompt_meta import build_non_answer_fields


def _extract_json_obj(text: str) -> Dict[str, object]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _ensure_json_keyword(prompt: str) -> str:
    if "json" in prompt.lower():
        return prompt
    return "Return valid json object.\n" + prompt


def _normalize_action_input(action_input: object) -> Dict[str, object]:
    if isinstance(action_input, dict):
        return action_input
    return {}


class StepReasoner:
    def __init__(self, llm: DeepSeekClient):
        self.llm = llm

    def plan_step(
        self,
        item: TimeSeriesExamItem,
        intent_task: str,
        evidence: Dict[str, object],
        gap: List[str],
        tools: List[str],
    ) -> Dict[str, object]:
        meta = build_non_answer_fields(item)
        prompt = (
            "You are a time-series reasoning controller. Decide ONLY the next best tool call.\n"
            "Return valid json object with keys: thought, action, action_input.\n"
            "JSON example: {\"thought\":\"...\",\"action\":\"tool_name\",\"action_input\":{\"series\":\"ts\"}}\n"
            f"Question: {item.question}\n"
            f"Non-answer fields: {json.dumps(meta, ensure_ascii=False)}\n"
            f"Task: {intent_task}\n"
            f"Gap predicates: {gap}\n"
            f"Current evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Allowed tools: {tools}\n"
            "For action_input, use a JSON object. Use series names ts, ts1, ts2 where needed.\n"
        )
        text = self.llm.generate(
            [{"role": "user", "content": _ensure_json_keyword(prompt)}],
            response_format={"type": "json_object"},
        )
        obj = _extract_json_obj(text)
        action = obj.get("action")
        if action in tools:
            obj["action_input"] = _normalize_action_input(obj.get("action_input"))
            return obj
        return self._fallback_plan(intent_task, tools)

    def choose_option(
        self,
        item: TimeSeriesExamItem,
        evidence: Dict[str, object],
    ) -> str:
        meta = build_non_answer_fields(item)
        prompt = (
            "Answer a multiple-choice question using evidence.\n"
            "Return valid json object only: {\"answer\": \"<exact option text>\"}\n"
            f"Question: {item.question}\n"
            f"Non-answer fields: {json.dumps(meta, ensure_ascii=False)}\n"
            f"Options: {json.dumps(item.options, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
            "The answer value must be exactly one of the options.\n"
        )
        text = self.llm.generate(
            [{"role": "user", "content": _ensure_json_keyword(prompt)}],
            response_format={"type": "json_object"},
        )
        obj = _extract_json_obj(text)
        ans = str(obj.get("answer", "")).strip()
        if ans in item.options:
            return ans
        return item.options[0]

    @staticmethod
    def _fallback_plan(intent_task: str, tools: List[str]) -> Dict[str, object]:
        pref = {
            "trend": "detect_trend",
            "seasonality": "seasonality",
            "anomaly": "zscore_anomaly",
            "noise": "autocorrelation",
            "stationarity": "is_stationary",
            "similarity": "pearson_corr",
            "causality": "granger_causality",
            "generic": "describe",
        }
        act = pref.get(intent_task, "describe")
        if act not in tools:
            act = tools[0]
        default_input: Dict[str, object]
        if act in {
            "pearson_corr",
            "spearman_corr",
            "shape_similarity",
            "euclidean_distance",
            "manhattan_distance",
            "dtw_distance",
            "cosine_similarity",
            "cross_correlation",
            "lagged_correlation",
            "granger_causality",
            "transfer_entropy",
        }:
            default_input = {"series1": "ts1", "series2": "ts2"}
        else:
            default_input = {"series": "ts"}
        return {"thought": "Fallback tool selection.", "action": act, "action_input": default_input}

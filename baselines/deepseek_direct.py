import json
from typing import Dict

from data.loader import TimeSeriesExamItem, series_payload
from agent.prompt_meta import build_non_answer_fields
from eval.judge import llm_judge_correctness
from llm.deepseek_client import DeepSeekClient


def _extract_answer(text: str, options: list[str]) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return options[0]
    ans = str(obj.get("answer", "")).strip()
    if ans in options:
        return ans
    return options[0]


def answer_one(client: DeepSeekClient, item: TimeSeriesExamItem) -> Dict[str, object]:
    payload = series_payload(item)
    meta = build_non_answer_fields(item)
    prompt = (
        "You are solving a multiple-choice time-series question.\n"
        "Return ONLY JSON with one key: answer.\n"
        "answer must be exactly one of the provided options.\n"
        f"Question: {item.question}\n"
        f"Options: {json.dumps(item.options, ensure_ascii=False)}\n"
        f"Series data: {json.dumps(payload, ensure_ascii=False)}\n"
        f"Non-answer fields: {json.dumps(meta, ensure_ascii=False)}\n"
        'Output format: {"answer":"<exact option text>"}'
    )
    raw = client.generate(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    pred = _extract_answer(raw, item.options)
    judged_correct, judge_reason = llm_judge_correctness(
        llm=client,
        question=item.question,
        non_answer_fields=meta,
        options=item.options,
        gold_answer=item.answer,
        pred_answer=pred,
    )
    return {
        "id": item.id,
        "question": item.question,
        "category": item.category,
        "subcategory": item.subcategory,
        "gold_answer": item.answer,
        "pred_option": pred,
        "is_correct": judged_correct,
        "judge_reason": judge_reason,
        "raw_response": raw,
    }

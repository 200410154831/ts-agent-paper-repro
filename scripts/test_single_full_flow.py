import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent.runner import TSAgentRunner
from data.loader import TimeSeriesExamItem, load_timeseries_exam
from llm.deepseek_client import DeepSeekClient


def _sanitize_prompt(text: str) -> str:
    # Hide raw time series payload from printed prompts.
    text = re.sub(r"(Time series data:\s*)(.*)", r"\1<REDACTED>", text)
    text = re.sub(r"(Series data:\s*)(.*)", r"\1<REDACTED>", text)
    return text


def _guess_stage(prompt: str) -> str:
    p = prompt.lower()
    if "intent extractor" in p:
        return "intent"
    if "reasoning controller" in p:
        return "reasoner_plan_step"
    if "critic for a time-series reasoning agent" in p:
        return "critic_feedback"
    if "final quality gate" in p:
        return "quality_gate"
    if "answer judge for multiple-choice time-series questions" in p:
        return "answer_judge"
    if "answer a multiple-choice question using evidence" in p:
        return "final_answer_generation"
    return "unknown"


class RecordingDeepSeekClient(DeepSeekClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_logs: List[Dict[str, Any]] = []

    def generate(self, messages: List[Dict[str, str]], response_format: Dict[str, Any] = None) -> str:
        prompt = messages[-1]["content"] if messages else ""
        response = super().generate(messages, response_format=response_format)
        self.call_logs.append(
            {
                "stage": _guess_stage(prompt),
                "prompt": _sanitize_prompt(prompt),
                "response_text": response,
                "response_json": _safe_json(response),
            }
        )
        return response


def _safe_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _pick_item(items: List[TimeSeriesExamItem], item_id: int, seed: int) -> TimeSeriesExamItem:
    if item_id > 0:
        for x in items:
            if x.id == item_id:
                return x
        raise ValueError(f"id={item_id} not found in dataset")
    random.seed(seed)
    return random.choice(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to TimeSeriesExam json file")
    parser.add_argument("--config", default="configs/deepseek_v3.yaml", help="DeepSeek config yaml")
    parser.add_argument("--api_key", default=None, help="Optional DeepSeek API key")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--id", type=int, default=0, help="Use a fixed item id instead of random sampling")
    parser.add_argument("--step_budget", type=int, default=5, help="TS-Agent step budget")
    parser.add_argument("--gate_max_calls", type=int, default=1, help="Maximum gate calls")
    parser.add_argument("--save_json", default="", help="Optional output file to save full debug json")
    args = parser.parse_args()

    items = load_timeseries_exam(args.data)
    item = _pick_item(items, args.id, args.seed)

    llm = RecordingDeepSeekClient.from_yaml(args.config, api_key=args.api_key)
    runner = TSAgentRunner(llm=llm, step_budget=args.step_budget, gate_max_calls=args.gate_max_calls)

    result = runner.run_item(item)

    print("=" * 90)
    print("SAMPLED ITEM")
    print("=" * 90)
    print(json.dumps(
        {
            "id": item.id,
            "category": item.category,
            "subcategory": item.subcategory,
            "question": item.question,
            "options": item.options,
            "gold_answer": item.answer,
        },
        ensure_ascii=False,
        indent=2,
    ))

    print("\n" + "=" * 90)
    print("ALL PROMPTS + MODEL JSON RESPONSES (TIME SERIES REDACTED)")
    print("=" * 90)
    for i, log in enumerate(llm.call_logs, start=1):
        print(f"\n--- CALL {i} | stage={log['stage']} ---")
        print("[PROMPT]")
        print(log["prompt"])
        print("\n[MODEL RESPONSE TEXT]")
        print(log["response_text"])
        print("\n[MODEL RESPONSE JSON]")
        print(json.dumps(log["response_json"], ensure_ascii=False, indent=2))

    print("\n" + "=" * 90)
    print("FINAL RESULT")
    print("=" * 90)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_json:
        payload = {
            "sampled_item": {
                "id": item.id,
                "category": item.category,
                "subcategory": item.subcategory,
                "question": item.question,
                "options": item.options,
                "gold_answer": item.answer,
            },
            "calls": llm.call_logs,
            "final_result": result,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved debug output to: {args.save_json}")


if __name__ == "__main__":
    main()

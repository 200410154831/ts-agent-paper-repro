import argparse
import json
import os
import random
import sys
from typing import Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent.runner import TSAgentRunner
from data.loader import load_timeseries_exam
from llm.deepseek_client import DeepSeekClient


def load_done_ids(path: str) -> Set[int]:
    if not os.path.exists(path):
        return set()
    done: Set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            done.add(int(obj["id"]))
    return done


def render_progress(prefix: str, done: int, total: int, current_id: int | None = None) -> None:
    width = 30
    ratio = (done / total) if total else 0.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    if current_id is None:
        msg = f"\r[{prefix}] [{bar}] {done}/{total} ({ratio * 100:5.1f}%)"
    else:
        msg = f"\r[{prefix}] [{bar}] {done}/{total} ({ratio * 100:5.1f}%) current_id={current_id}"
    print(msg, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/deepseek_v3.yaml")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--step_budget", type=int, default=5)
    parser.add_argument("--gate_max_calls", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--sample_ids_file", default="", help="Optional JSON file of sampled ids to run")
    parser.add_argument("--sample_n", type=int, default=0, help="Randomly sample N items from dataset")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for sample_n")
    parser.add_argument("--save_sample_ids", default="", help="Optional output JSON file for sampled ids")
    args = parser.parse_args()

    client = DeepSeekClient.from_yaml(args.config, api_key=args.api_key)
    runner = TSAgentRunner(client, step_budget=args.step_budget, gate_max_calls=args.gate_max_calls)
    items = load_timeseries_exam(args.data)
    if args.sample_ids_file:
        with open(args.sample_ids_file, "r", encoding="utf-8") as f:
            selected_ids = set(int(x) for x in json.load(f))
        items = [x for x in items if x.id in selected_ids]
    elif args.sample_n and args.sample_n > 0:
        random.seed(args.sample_seed)
        sample_n = min(args.sample_n, len(items))
        items = random.sample(items, sample_n)
        if args.save_sample_ids:
            os.makedirs(os.path.dirname(args.save_sample_ids), exist_ok=True)
            with open(args.save_sample_ids, "w", encoding="utf-8") as f:
                json.dump(sorted([x.id for x in items]), f, ensure_ascii=False, indent=2)
    done = load_done_ids(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total_target = len(items)
    print(f"[agent] total_items={total_target}, already_done={len(done)}, to_run={max(0, total_target - len(done))}")
    render_progress("agent", len(done), total_target)

    with open(args.output, "a", encoding="utf-8") as f:
        processed = 0
        for idx, item in enumerate(items, start=1):
            if item.id in done:
                continue
            if args.max_items and processed >= args.max_items:
                break
            try:
                result = runner.run_item(item)
                result["status"] = "ok"
            except Exception as e:
                result = {
                    "id": item.id,
                    "question": item.question,
                    "category": item.category,
                    "subcategory": item.subcategory,
                    "gold_answer": item.answer,
                    "pred_option": item.options[0],
                    "is_correct": item.options[0] == item.answer,
                    "trace": [],
                    "evidence": {},
                    "quality_gate_passed": False,
                    "quality_gate_reasons": ["runtime_error"],
                    "status": "error",
                    "error": str(e),
                }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            processed += 1
            done_now = len(done) + processed
            render_progress("agent", done_now, total_target, current_id=item.id)
    print()
    print(f"[agent] completed. total_written={processed}, final_done={len(done) + processed}/{total_target}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from typing import Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from baselines.deepseek_direct import answer_one
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
    parser.add_argument("--max_items", type=int, default=0)
    args = parser.parse_args()

    client = DeepSeekClient.from_yaml(args.config, api_key=args.api_key)
    items = load_timeseries_exam(args.data)
    done = load_done_ids(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total_target = len(items)
    print(f"[baseline] total_items={total_target}, already_done={len(done)}, to_run={max(0, total_target - len(done))}")
    render_progress("baseline", len(done), total_target)

    with open(args.output, "a", encoding="utf-8") as f:
        processed = 0
        for idx, item in enumerate(items, start=1):
            if item.id in done:
                continue
            if args.max_items and processed >= args.max_items:
                break
            try:
                result = answer_one(client, item)
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
                    "raw_response": "",
                    "status": "error",
                    "error": str(e),
                }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            processed += 1
            done_now = len(done) + processed
            render_progress("baseline", done_now, total_target, current_id=item.id)
    print()
    print(f"[baseline] completed. total_written={processed}, final_done={len(done) + processed}/{total_target}")


if __name__ == "__main__":
    main()

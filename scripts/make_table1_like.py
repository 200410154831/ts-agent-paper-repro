import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval.report_table import build_table_rows, to_markdown
from eval.scoring import category_accuracy, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline)
    agent_rows = load_jsonl(args.agent)
    baseline_rows = [r for r in baseline_rows if r.get("status") == "ok"]
    agent_rows = [r for r in agent_rows if r.get("status") == "ok"]

    base_acc = category_accuracy(baseline_rows)
    agent_acc = category_accuracy(agent_rows)
    df = build_table_rows(base_acc, agent_acc)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(to_markdown(df) + "\n")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

# TS-Agent Reproduction with DeepSeek-V3

This project reproduces a TS-Agent-style evaluation on TimeSeriesExam with two settings:

1. DeepSeek-V3 direct answering (baseline)
2. DeepSeek-V3 with an evidence-driven TS-Agent loop (reasoner + tools + critic + quality gate)

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set API key:

```bash
export DEEPSEEK_API_KEY="your_key"
```

3. Run baseline:

```bash
python scripts/run_baseline.py \
  --data /root/autodl-tmp/data/TimeSeriesExam_round3_qa_dataset.json \
  --output results/baseline_predictions.jsonl
```

4. Run agent:

```bash
python scripts/run_agent.py \
  --data /root/autodl-tmp/data/TimeSeriesExam_round3_qa_dataset.json \
  --output results/agent_predictions.jsonl
```

5. Build table:

```bash
python scripts/make_table1_like.py \
  --baseline results/baseline_predictions.jsonl \
  --agent results/agent_predictions.jsonl \
  --out_csv results/table1_like.csv \
  --out_md results/table1_like.md
```

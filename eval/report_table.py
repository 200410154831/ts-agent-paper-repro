from typing import Dict

import pandas as pd

from eval.scoring import TABLE1_ORDER


def build_table_rows(baseline: Dict[str, float], agent: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for label, source in [("DeepSeek-V3", baseline), ("DeepSeek-V3 + TS-Agent", agent)]:
        rows.append(
            {
                "Model": label,
                "Pattern Rec.": round(source["Pattern Recognition"], 4),
                "Noise Und.": round(source["Noise Understanding"], 4),
                "Anomaly Det.": round(source["Anomaly Detection"], 4),
                "Similarity": round(source["Similarity Analysis"], 4),
                "Causality": round(source["Causality Analysis"], 4),
                "Overall": round(source["Overall"], 4),
                "Total": int(source["Total"]),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)

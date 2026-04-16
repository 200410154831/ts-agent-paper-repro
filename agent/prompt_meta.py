from typing import Any, Dict

from data.loader import TimeSeriesExamItem


def build_non_answer_fields(item: TimeSeriesExamItem) -> Dict[str, Any]:
    return {
        "category": item.raw.get("category"),
        "subcategory": item.raw.get("subcategory"),
        "question_type": item.raw.get("question_type"),
        "format_hint": item.raw.get("format_hint"),
        "question_hint": item.raw.get("question_hint"),
        "relevant_concepts": item.raw.get("relevant_concepts"),
    }

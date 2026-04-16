from typing import Any, Dict, List

from agent.critic import review_step
from agent.intent import detect_intent, update_covered_predicates
from agent.quality_gate import check_final_answer
from agent.prompt_meta import build_non_answer_fields
from agent.root_tools_registry import RootToolsRegistry
from agent.reasoner import StepReasoner
from data.loader import TimeSeriesExamItem
from eval.judge import llm_judge_correctness
from llm.deepseek_client import DeepSeekClient


def _update_evidence(evidence: Dict[str, Any], intent_task: str, obs: Dict[str, Any]) -> None:
    if "error" in obs:
        return
    result = obs.get("result", {})
    if not isinstance(result, dict):
        result = {"data": result}
    data = result.get("data")

    if intent_task == "trend":
        evidence["trend_direction"] = data
    elif intent_task == "seasonality":
        evidence["seasonality"] = data
    elif intent_task == "anomaly":
        if isinstance(data, list):
            anomaly_indices = [i for i, v in enumerate(data) if int(v) == 1]
            evidence["has_anomaly"] = len(anomaly_indices) > 0
            evidence["anomaly_segment_or_type"] = anomaly_indices
        else:
            evidence["has_anomaly"] = bool(data)
            evidence["anomaly_segment_or_type"] = data
    elif intent_task == "noise":
        if isinstance(data, (int, float)):
            evidence["noise_type"] = "white" if abs(float(data)) < 0.2 else "red"
        else:
            evidence["noise_type"] = data
    elif intent_task == "stationarity":
        if isinstance(data, dict):
            evidence["stationary_flag"] = data.get("is_stationary")
        elif isinstance(data, str):
            evidence["stationary_flag"] = data.lower() == "stationary"
        else:
            evidence["stationary_flag"] = bool(data)
    elif intent_task in {"similarity", "causality"}:
        evidence["corr_or_distance"] = data
        evidence["lead_relation"] = data
    else:
        evidence["basic_stats"] = data


class TSAgentRunner:
    def __init__(self, llm: DeepSeekClient, step_budget: int = 5, gate_max_calls: int = 1):
        self.reasoner = StepReasoner(llm)
        self.step_budget = step_budget
        self.gate_max_calls = gate_max_calls
        self.registry = RootToolsRegistry()
        self.tools = self.registry.tool_names

    def run_item(self, item: TimeSeriesExamItem) -> Dict[str, Any]:
        meta = build_non_answer_fields(item)
        intent = detect_intent(item, self.reasoner.llm)
        evidence: Dict[str, Any] = {}
        trace: List[Dict[str, Any]] = []
        gate_history: List[Dict[str, Any]] = []
        gate_calls = 0
        current_step_num = 0

        def recompute_gap() -> List[str]:
            covered_local = update_covered_predicates(evidence, intent.required_predicates)
            return [x for x in intent.required_predicates if x not in covered_local]

        gap = recompute_gap()
        gate_ok = False
        reasons: List[str] = []
        pred = item.options[0]

        while current_step_num < self.step_budget:
            plan = self.reasoner.plan_step(item, intent.task, evidence, gap, self.tools)
            action = str(plan.get("action", "summary_stats"))
            action_input = plan.get("action_input", {"series": "ts"})
            if not isinstance(action_input, dict):
                action_input = {"series": "ts"}
            obs = self.registry.execute(item, action, action_input)
            _update_evidence(evidence, intent.task, obs)

            gap = recompute_gap()
            critic = review_step(
                llm=self.reasoner.llm,
                question=item.question,
                non_answer_fields=meta,
                action=action,
                action_input=action_input,
                observation=obs,
                evidence=evidence,
                gap=gap,
            )

            trace.append(
                {
                    "step": current_step_num + 1,
                    "thought": plan.get("thought", ""),
                    "action": action,
                    "action_input": action_input,
                    "observation": obs,
                    "critic_feedback": critic,
                    "gap": gap,
                }
            )
            current_step_num += 1

            if not gap:
                pred = self.reasoner.choose_option(item, evidence)
                critic_enough = critic.get("evidence_sufficiency") == "enough"

                if critic_enough and gate_calls >= self.gate_max_calls:
                    gate_ok = True
                    reasons = ["skip_gate_due_to_gate_limit_and_sufficient_evidence"]
                    gate_history.append(
                        {
                            "at_step": current_step_num,
                            "gate_call_index": gate_calls,
                            "pred_before_gate": pred,
                            "gate_ok": gate_ok,
                            "reasons": reasons,
                            "backtrack_step": None,
                        }
                    )
                    break

                gate_ok, reasons, backtrack_step = check_final_answer(
                    llm=self.reasoner.llm,
                    question=item.question,
                    non_answer_fields=meta,
                    pred=pred,
                    options=item.options,
                    gap=gap,
                    evidence=evidence,
                )
                gate_calls += 1
                gate_history.append(
                    {
                        "at_step": current_step_num,
                        "gate_call_index": gate_calls,
                        "pred_before_gate": pred,
                        "gate_ok": gate_ok,
                        "reasons": reasons,
                        "backtrack_step": backtrack_step,
                    }
                )
                if gate_ok:
                    break

                # gate rejected -> backtrack and continue from that step
                if backtrack_step is not None and 1 <= backtrack_step <= len(trace):
                    trace = trace[:backtrack_step]
                    evidence = {}
                    for t in trace:
                        _update_evidence(evidence, intent.task, t.get("observation", {}))
                    gap = recompute_gap()
                    current_step_num = len(trace)
                    continue

                # no valid backtrack step; continue stepping if budget allows
                gap = recompute_gap()

        if not gate_ok:
            pred = item.options[0]
            if not reasons:
                reasons = ["step_budget_reached_or_gate_rejected"]

        judged_correct, judge_reason = llm_judge_correctness(
            llm=self.reasoner.llm,
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
            "trace": trace,
            "evidence": evidence,
            "quality_gate_passed": gate_ok,
            "quality_gate_reasons": reasons,
            "gate_history": gate_history,
            "gate_calls": gate_calls,
            "step_count": current_step_num,
            "step_budget": self.step_budget,
            "step_budget_reached": current_step_num >= self.step_budget,
        }

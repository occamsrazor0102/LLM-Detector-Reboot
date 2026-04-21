"""Robustness evaluation: measure pipeline performance delta under adversarial attack.

Given a clean labeled dataset and a list of attacks, generate adversarial
variants of the LLM-labeled samples, run the pipeline on both, and compute
per-attack metric deltas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from beet.adversarial.generator import generate
from beet.evaluation.dataset import EvalSample
from beet.evaluation.runner import EvalReport, run_eval


@dataclass(frozen=True)
class RobustnessReport:
    baseline: EvalReport
    per_attack: dict[str, EvalReport]
    attack_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    vulnerability_ranking: list[tuple[str, float]] = field(default_factory=list)


def _delta(attack_metrics: dict, baseline_metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, base_val in baseline_metrics.items():
        try:
            out[key] = float(attack_metrics[key]) - float(base_val)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def run_robustness_eval(
    pipeline,
    clean_dataset: list[EvalSample],
    attacks: list[str],
    *,
    provider: Callable | None = None,
    seed: int = 42,
    progress: bool = False,
) -> RobustnessReport:
    baseline = run_eval(pipeline, clean_dataset, progress=progress)

    human_samples = [s for s in clean_dataset if s.label == 0]
    per_attack: dict[str, EvalReport] = {}
    attack_deltas: dict[str, dict[str, float]] = {}

    for attack_name in attacks:
        adv = generate(clean_dataset, [attack_name], provider=provider, seed=seed)
        combined = human_samples + adv
        if not combined:
            continue
        report = run_eval(pipeline, combined, progress=progress)
        per_attack[attack_name] = report
        attack_deltas[attack_name] = _delta(report.metrics, baseline.metrics)

    # Rank by AUROC degradation (most negative delta = most harmful attack)
    ranked = [
        (name, d.get("auroc", 0.0))
        for name, d in attack_deltas.items()
    ]
    ranked.sort(key=lambda x: x[1])

    return RobustnessReport(
        baseline=baseline,
        per_attack=per_attack,
        attack_deltas=attack_deltas,
        vulnerability_ranking=ranked,
    )

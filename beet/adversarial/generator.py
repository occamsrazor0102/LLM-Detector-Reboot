"""Adversarial sample generator: source dataset + attacks -> adversarial dataset."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from beet.adversarial.registry import get_attack
from beet.evaluation.dataset import EvalSample


def generate(
    source_dataset: list[EvalSample],
    attacks: list[str],
    *,
    provider: Callable | None = None,
    seed: int | None = None,
) -> list[EvalSample]:
    """Apply each attack to every LLM-labeled sample and return the resulting samples.

    Human samples (label == 0) are skipped — transforms on human text are meaningless.
    Prompt-category attacks require `provider`; transform attacks ignore it.
    """
    out: list[EvalSample] = []
    for source in source_dataset:
        if source.label != 1:
            continue
        for attack_name in attacks:
            attack = get_attack(attack_name)
            if attack is None:
                raise ValueError(f"unknown attack: {attack_name}")
            kwargs = {}
            if seed is not None:
                kwargs["seed"] = seed
            if attack.category == "prompt":
                if provider is None:
                    raise ValueError(
                        f"attack {attack_name!r} is prompt-based and requires a provider callable"
                    )
                kwargs["provider"] = provider
            new_text = attack.apply(source.text, **kwargs)
            out.append(replace(
                source,
                id=f"{source.id}__{attack_name}",
                text=new_text,
                attack_name=attack_name,
                attack_category=attack.category,
                source_id=source.id,
            ))
    return out

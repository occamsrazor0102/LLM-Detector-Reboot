"""Attack registry for adversarial module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Attack:
    name: str
    category: str  # "transform" | "prompt"
    description: str
    severity: str  # "basic" | "moderate" | "advanced"
    apply: Callable[..., str] = None

    def __call__(self, text: str, **kwargs) -> str:
        if self.apply is None:
            raise NotImplementedError(f"Attack {self.name} has no apply fn")
        return self.apply(text, **kwargs)


_attacks: dict[str, Attack] = {}


def register(attack: Attack) -> Attack:
    _attacks[attack.name] = attack
    return attack


def get_attack(name: str) -> Attack | None:
    return _attacks.get(name)


def list_attacks(category: str | None = None) -> list[Attack]:
    items = list(_attacks.values())
    if category is not None:
        items = [a for a in items if a.category == category]
    return sorted(items, key=lambda a: (a.category, a.name))

"""BEET 2.0 adversarial testing — generate evasion samples for stress-testing."""
from beet.adversarial.registry import Attack, register, get_attack, list_attacks
from beet.adversarial.generator import generate
from beet.adversarial import transforms as _transforms  # noqa: F401 — registers attacks

__all__ = ["Attack", "register", "get_attack", "list_attacks", "generate"]

from beet.adversarial import list_attacks, get_attack, Attack
import beet.adversarial  # triggers registration


class TestRegistry:
    def test_builtin_transforms_registered(self):
        names = {a.name for a in list_attacks(category="transform")}
        assert {"strip_preamble", "inject_typos", "casualize", "synonym_swap", "mix_human"} <= names

    def test_filter_by_category(self):
        transforms = list_attacks(category="transform")
        assert all(a.category == "transform" for a in transforms)

    def test_get_missing_returns_none(self):
        assert get_attack("nonexistent_attack") is None

    def test_get_found(self):
        a = get_attack("strip_preamble")
        assert isinstance(a, Attack)
        assert a.severity == "basic"

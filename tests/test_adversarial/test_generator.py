import pytest
from beet.evaluation.dataset import EvalSample
from beet.adversarial.generator import generate
import beet.adversarial  # registers attacks


@pytest.fixture
def source_dataset():
    return [
        EvalSample(id="h1", text="Human text.", label=0, tier="human"),
        EvalSample(id="l1", text="Certainly! Here is comprehensive output.", label=1, tier="A0"),
        EvalSample(id="l2", text="Furthermore, we utilize many approaches.", label=1, tier="A0"),
    ]


class TestGenerate:
    def test_skips_human_samples(self, source_dataset):
        out = generate(source_dataset, ["strip_preamble"])
        assert all(s.label == 1 for s in out)
        assert not any(s.source_id == "h1" for s in out)

    def test_cardinality_is_sources_times_attacks(self, source_dataset):
        # 2 LLM sources * 2 attacks = 4
        out = generate(source_dataset, ["strip_preamble", "synonym_swap"])
        assert len(out) == 4

    def test_id_and_fields_set(self, source_dataset):
        out = generate(source_dataset, ["strip_preamble"])
        s = out[0]
        assert s.id.endswith("__strip_preamble")
        assert s.attack_name == "strip_preamble"
        assert s.attack_category == "transform"
        assert s.source_id in {"l1", "l2"}

    def test_unknown_attack_raises(self, source_dataset):
        with pytest.raises(ValueError, match="unknown attack"):
            generate(source_dataset, ["no_such_attack"])

    def test_text_modified(self, source_dataset):
        out = generate(source_dataset, ["synonym_swap"])
        # l2 mentions "Furthermore" and "utilize" — both get swapped
        swapped = next(s for s in out if s.source_id == "l2")
        assert "Furthermore" not in swapped.text
        assert "utilize" not in swapped.text

import json
import pytest
from pathlib import Path
from beet.evaluation.dataset import EvalSample, load_dataset, save_dataset, build_dataset

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

class TestLoadDataset:
    def test_loads_eval_mini(self):
        samples = load_dataset(FIXTURES_DIR / "eval_mini.jsonl")
        assert len(samples) >= 6
        assert all(isinstance(s, EvalSample) for s in samples)
        assert all(isinstance(s.label, int) for s in samples)

    def test_malformed_missing_label(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"id": "x", "text": "hello"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 1.*missing required field 'label'"):
            load_dataset(bad)

    def test_malformed_bad_json(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('not json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 1.*invalid JSON"):
            load_dataset(bad)

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "gaps.jsonl"
        f.write_text(
            '{"id":"a","text":"hello","label":0}\n'
            '\n'
            '{"id":"b","text":"world","label":1}\n',
            encoding="utf-8",
        )
        assert len(load_dataset(f)) == 2

    def test_unknown_fields_ignored(self, tmp_path):
        f = tmp_path / "extra.jsonl"
        f.write_text('{"id":"a","text":"hi","label":0,"extra_field":"ignored"}\n', encoding="utf-8")
        samples = load_dataset(f)
        assert len(samples) == 1
        assert samples[0].id == "a"


class TestSaveDataset:
    def test_roundtrip(self, tmp_path):
        original = [
            EvalSample(id="s0", text="hello", label=0, tier="human"),
            EvalSample(id="s1", text="world", label=1, tier="A0", source="gpt4"),
        ]
        path = tmp_path / "out.jsonl"
        save_dataset(original, path)
        loaded = load_dataset(path)
        assert loaded == original

    def test_omits_none_fields(self, tmp_path):
        samples = [EvalSample(id="s0", text="hi", label=0)]
        path = tmp_path / "out.jsonl"
        save_dataset(samples, path)
        raw = json.loads(path.read_text(encoding="utf-8").strip())
        assert "tier" not in raw
        assert "source" not in raw
        assert "attack_name" not in raw


class TestBuildDataset:
    def test_from_directory(self, tmp_path):
        d = tmp_path / "texts"
        d.mkdir()
        (d / "a.txt").write_text("first", encoding="utf-8")
        (d / "b.txt").write_text("second", encoding="utf-8")
        samples = build_dataset([{"path": str(d), "label": 1, "tier": "A0"}])
        assert len(samples) == 2
        assert samples[0].id == "A0_0000"
        assert samples[1].id == "A0_0001"
        assert samples[0].text == "first"
        assert all(s.label == 1 for s in samples)

    def test_from_single_file(self, tmp_path):
        f = tmp_path / "one.txt"
        f.write_text("content", encoding="utf-8")
        samples = build_dataset([{"path": str(f), "label": 0}])
        assert len(samples) == 1
        assert samples[0].id == "sample_0000"

    def test_id_prefix(self, tmp_path):
        f = tmp_path / "one.txt"
        f.write_text("content", encoding="utf-8")
        samples = build_dataset([{"path": str(f), "label": 0, "tier": "human"}], id_prefix="test_")
        assert samples[0].id == "test_human_0000"

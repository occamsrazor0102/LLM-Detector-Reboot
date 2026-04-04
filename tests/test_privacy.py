from beet.privacy.hashing import hash_text, normalize_for_hash
from beet.privacy.store import PrivacyStore

def test_hash_is_deterministic():
    assert hash_text("hello world") == hash_text("hello world")

def test_hash_is_different_for_different_texts():
    assert hash_text("hello") != hash_text("world")

def test_hash_format():
    h = hash_text("test")
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64

def test_normalize_for_hash_strips_whitespace():
    assert normalize_for_hash("  hello  world  ") == normalize_for_hash("hello world")

def test_store_saves_feature_vector(tmp_path):
    store = PrivacyStore(base_dir=tmp_path)
    store.save_features("sub_001", {"fingerprint_hits": 5.0, "binoculars_ratio": 0.92}, "sha256:abc123", "AMBER")
    loaded = store.get_features("sub_001")
    assert loaded is not None
    assert loaded["feature_vector"]["fingerprint_hits"] == 5.0

def test_store_does_not_save_raw_text_by_default(tmp_path):
    store = PrivacyStore(base_dir=tmp_path)
    store.save_features("sub_002", {}, "sha256:def456", "GREEN")
    assert not (tmp_path / "raw_text_vault" / "sub_002.txt").exists()

def test_store_can_save_raw_text_with_explicit_flag(tmp_path):
    store = PrivacyStore(base_dir=tmp_path)
    store.save_raw_text("sub_003", "This is the original text.", reason="human_review_requested")
    assert (tmp_path / "raw_text_vault" / "sub_003.json").exists()

"""Config-redaction scope tests — not a FastAPI integration test, just the
pure helper so they run regardless of whether `beet[api]` extras are
installed."""
import pytest

pytest.importorskip("fastapi", reason="api helper imports fastapi-dependent types")

from beet.api import _redact


def test_redact_scrubs_top_level_sensitive_keys():
    cfg = {"api_key": "sk-live-abcdef", "other": "ok"}
    out = _redact(cfg)
    assert out["api_key"] == "[redacted]"
    assert out["other"] == "ok"


def test_redact_walks_nested_dicts():
    cfg = {"tier3": {"anthropic": {"api_key": "sk-ant-x", "model": "claude"}}}
    out = _redact(cfg)
    assert out["tier3"]["anthropic"]["api_key"] == "[redacted]"
    assert out["tier3"]["anthropic"]["model"] == "claude"


def test_redact_catches_all_documented_tokens():
    cfg = {
        "password": "p1", "auth_token": "t1", "credentials": {"user": "u"},
        "auth": "bearer x", "secret_sauce": "s", "not_sensitive": 1,
    }
    out = _redact(cfg)
    for k in ("password", "auth_token", "credentials", "auth", "secret_sauce"):
        assert out[k] == "[redacted]", f"expected {k!r} to be redacted"
    assert out["not_sensitive"] == 1


def test_redact_preserves_lists_but_descends_into_dict_items():
    cfg = {
        "providers": [
            {"name": "anthropic", "api_key": "x"},
            {"name": "openai", "api_key": "y"},
        ],
        "port": 8000,
    }
    out = _redact(cfg)
    assert out["providers"][0]["api_key"] == "[redacted]"
    assert out["providers"][1]["api_key"] == "[redacted]"
    assert out["providers"][0]["name"] == "anthropic"
    assert out["port"] == 8000

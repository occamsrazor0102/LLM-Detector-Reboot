from beet.normalizer import normalize_text

def test_removes_invisible_characters():
    text = "hello\u200bworld"
    assert "\u200b" not in normalize_text(text)

def test_normalizes_nfkc():
    text = "ＡＢＣ"
    result = normalize_text(text)
    assert result == "ABC"

def test_collapses_extra_whitespace():
    text = "hello   world\n\n\nfoo"
    result = normalize_text(text)
    assert "   " not in result

def test_replaces_common_homoglyphs():
    text = "d\u0435lve"
    result = normalize_text(text)
    assert "\u0435" not in result

def test_returns_string():
    assert isinstance(normalize_text("hello"), str)

def test_empty_string():
    assert normalize_text("") == ""

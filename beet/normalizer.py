import re
import unicodedata

_HOMOGLYPHS: dict[int, int] = {
    0x0430: ord("a"), 0x0435: ord("e"), 0x043E: ord("o"),
    0x0440: ord("r"), 0x0441: ord("c"), 0x0445: ord("x"),
    0x0443: ord("y"), 0x03BF: ord("o"), 0x03B1: ord("a"),
    0x00E0: ord("a"),
}
_INVISIBLE = re.compile(r"[\u00AD\u200B\u200C\u200D\u2060\uFEFF\u00A0\u2028\u2029]")

def normalize_text(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HOMOGLYPHS)
    text = _INVISIBLE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

import hashlib
import re
import unicodedata

def normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def hash_text(text: str) -> str:
    normalized = normalize_for_hash(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

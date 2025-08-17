import re
from typing import Dict, List, Tuple, Optional, Set

from .data import PolicyRecord

_STOP_WORDS_CACHE: Optional[Set[str]] = None

def _get_stop_words() -> Set[str]:
    """Lazily load NLTK stop words; fall back to empty set if unavailable.

    Avoids importing/loading large NLTK corpora at module import time which can
    slow down or block if the corpus isn't present.
    """
    global _STOP_WORDS_CACHE
    if _STOP_WORDS_CACHE is not None:
        return _STOP_WORDS_CACHE
    try:
        from nltk.corpus import stopwords  # type: ignore
        _STOP_WORDS_CACHE = set(stopwords.words("english"))
    except Exception:
        _STOP_WORDS_CACHE = set()
    return _STOP_WORDS_CACHE


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()
    return text

def preprocess_records(records: List[PolicyRecord]) -> List[PolicyRecord]:
    """Lightweight per-record preprocessing optimized for speed.

    - Normalize case and strip HTML/non-alphanum characters
    - Remove stopwords if available

    Note: We intentionally avoid stemming and lemmatization here to preserve
    token shapes for NER span alignment and to keep preprocessing fast.
    """
    stop_words = _get_stop_words()
    for rec in records:
        normalized = preprocess_text(rec.policy_text)
        if stop_words:
            tokens = normalized.split()
            normalized = " ".join(t for t in tokens if t not in stop_words)
        rec.policy_text = normalized
    return records


def to_bio_tags(text: str, entities: Dict[str, List[Dict]]) -> Tuple[List[str], List[str]]:
    tokens = text.split()
    tags = ["O"] * len(tokens)
    # Build character index to token index mapping
    char_to_tok = {}
    idx = 0
    for ti, tok in enumerate(tokens):
        for _ in tok:
            char_to_tok[idx] = ti
            idx += 1
        # account for the space
        char_to_tok[idx] = ti
        idx += 1

    for label, spans in entities.items():
        if not isinstance(spans, list):
            continue
        for span in spans:
            try:
                start = int(span.get("start"))
                end = int(span.get("end"))
            except Exception:
                continue
            if start not in char_to_tok or (end - 1) not in char_to_tok:
                continue
            start_tok = char_to_tok[start]
            end_tok = char_to_tok[end - 1]
            tags[start_tok] = f"B-{label}"
            for ti in range(start_tok + 1, end_tok + 1):
                tags[ti] = f"I-{label}"

    return tokens, tags


import importlib
import re
from typing import Any, Optional

_spacy_module: Optional[Any] = None
_nlp = None


def get_spacy_nlp():
    global _nlp
    global _spacy_module
    if _spacy_module is None:
        try:
            _spacy_module = importlib.import_module("spacy")
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("spaCy is not installed") from exc
    if _nlp is None:
        try:
            _nlp = _spacy_module.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        except Exception:
            _nlp = _spacy_module.blank("en")
    return _nlp


def _fallback_tokenize(text: str):
    if not text:
        return None, [], []
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\w+|[^\w\s]", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return None, tokens, offsets


def tokenize_with_doc(text: str):
    if not text:
        return None, [], []
    try:
        nlp = get_spacy_nlp()
        doc = nlp(text)
        tokens = [token.text for token in doc]
        offsets = [(token.idx, token.idx + len(token.text)) for token in doc]
        return doc, tokens, offsets
    except Exception:
        return _fallback_tokenize(text)

"""
Text preprocessing for PubGuard.

Designed for text *already extracted from PDFs* (e.g. via pdfplumber,
PyMuPDF, or GROBID in the PubVerse pipeline).  Focuses on cleaning
OCR / layout artefacts and producing a compact representation that
captures enough signal for the three classification heads.
"""

import re
from typing import Optional

# ── Compiled patterns ────────────────────────────────────────────

_WHITESPACE  = re.compile(r"\s+")
_HEADER_JUNK = re.compile(
    r"(doi:\s*\S+|https?://\S+|©\s*\d{4}|all rights reserved)",
    re.IGNORECASE,
)
_PAGE_NUMBER = re.compile(r"\n\s*\d{1,4}\s*\n")
_LIGATURE    = re.compile(r"[ﬁﬂﬀﬃﬄ]")

# Structural markers we look for to characterise document type
SECTION_HEADINGS = re.compile(
    r"\b(abstract|introduction|methods?|methodology|results|discussion|"
    r"conclusions?|references|bibliography|acknowledgments?|funding|"
    r"supplementary|materials?\s+and\s+methods?|related\s+work|"
    r"background|literature\s+review|experimental|data\s+availability)\b",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(
    r"\[\d+\]|\(\w+\s+et\s+al\.\s*,?\s*\d{4}\)|\(\w+,\s*\d{4}\)",
)


def clean_text(text: Optional[str], max_chars: int = 4000) -> str:
    """
    Normalise raw PDF-extracted text for embedding.

    Steps:
        1. Replace ligatures with ASCII equivalents.
        2. Strip DOIs, URLs, copyright lines.
        3. Remove isolated page numbers.
        4. Collapse whitespace.
        5. Truncate to `max_chars`.
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Ligatures
    text = _LIGATURE.sub(lambda m: {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"
    }.get(m.group(), m.group()), text)

    text = _HEADER_JUNK.sub(" ", text)
    text = _PAGE_NUMBER.sub("\n", text)
    text = _WHITESPACE.sub(" ", text).strip()

    return text[:max_chars]


def extract_structural_features(text: str) -> dict:
    """
    Cheap heuristic features that augment the embedding signal.

    Returns a dict of float features (0-1 range) that the linear
    head can concatenate with the embedding vector.
    """
    if not text:
        return _empty_features()

    n_chars = len(text)
    n_words = len(text.split())

    # Section heading density
    headings = SECTION_HEADINGS.findall(text)
    unique_headings = set(h.lower() for h in headings)

    # Citation density
    citations = CITATION_PATTERN.findall(text)

    # Character-level ratios
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    upper = sum(c.isupper() for c in text)

    return {
        # Document length signals (log-scaled, clipped)
        "log_chars": min(1.0, len(text) / 4000),
        "log_words": min(1.0, n_words / 800),

        # Structure signals
        "n_unique_sections": min(1.0, len(unique_headings) / 8),
        "has_abstract": float("abstract" in unique_headings),
        "has_methods": float(bool(unique_headings & {"methods", "methodology", "materials and methods"})),
        "has_references": float(bool(unique_headings & {"references", "bibliography"})),
        "has_introduction": float("introduction" in unique_headings),
        "has_results": float("results" in unique_headings),
        "has_discussion": float("discussion" in unique_headings),

        # Citation density
        "citation_density": min(1.0, len(citations) / max(n_words, 1) * 100),

        # Character composition
        "alpha_ratio": alpha / max(n_chars, 1),
        "digit_ratio": digit / max(n_chars, 1),
        "upper_ratio": upper / max(alpha, 1),

        # Mean sentence length (proxy for formality)
        "mean_sentence_len": min(1.0, _mean_sentence_length(text) / 50),
    }


def _mean_sentence_length(text: str) -> float:
    """Average words per sentence (rough split on .!?)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def _empty_features() -> dict:
    return {
        "log_chars": 0.0, "log_words": 0.0,
        "n_unique_sections": 0.0,
        "has_abstract": 0.0, "has_methods": 0.0,
        "has_references": 0.0, "has_introduction": 0.0,
        "has_results": 0.0, "has_discussion": 0.0,
        "citation_density": 0.0,
        "alpha_ratio": 0.0, "digit_ratio": 0.0, "upper_ratio": 0.0,
        "mean_sentence_len": 0.0,
    }


STRUCTURAL_FEATURE_NAMES = list(_empty_features().keys())
N_STRUCTURAL_FEATURES = len(STRUCTURAL_FEATURE_NAMES)

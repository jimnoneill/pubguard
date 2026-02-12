"""
PubGuard — Scientific Publication Gatekeeper
=============================================

Multi-head document classifier for the PubVerse pipeline.
Determines whether extracted PDF text represents a genuine
scientific publication vs. junk, and flags AI-generated or
offensive content.

Classification heads:
    1. doc_type   – scientific_paper | poster | abstract_only | junk
    2. ai_detect  – human | ai_generated
    3. toxicity   – clean | toxic

Architecture mirrors openalex-topic-classifier:
    model2vec (StaticModel) → L2-normalised embeddings → per-head
    linear classifiers (sklearn / small torch heads) stored as
    numpy weight matrices for zero-dependency inference.

Usage:
    from pubguard import PubGuard

    guard = PubGuard()
    guard.initialize()
    verdict = guard.screen(text)
    # verdict = {
    #   'doc_type': {'label': 'scientific_paper', 'score': 0.94},
    #   'ai_generated': {'label': 'human', 'score': 0.87},
    #   'toxicity': {'label': 'clean', 'score': 0.99},
    #   'pass': True
    # }
"""

from .classifier import PubGuard
from .config import PubGuardConfig
from .errors import (
    PubVerseError,
    build_pubguard_error,
    empty_input_error,
    unreadable_pdf_error,
    models_missing_error,
    gate_bypassed,
    format_error_line,
    PIPELINE_ERRORS,
)

__version__ = "0.1.0"
__all__ = [
    "PubGuard",
    "PubGuardConfig",
    "PubVerseError",
    "build_pubguard_error",
    "empty_input_error",
    "unreadable_pdf_error",
    "models_missing_error",
    "gate_bypassed",
    "format_error_line",
    "PIPELINE_ERRORS",
]

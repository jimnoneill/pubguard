"""
Configuration for PubGuard classifier.

Mirrors openalex_classifier.config with multi-head additions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os


def _find_models_dir() -> Path:
    """Locate PubGuard models directory.

    Checks for 'head_doc_type.npz' to distinguish PubGuard models
    from other model directories (e.g. OpenAlex) that may exist nearby.
    """
    marker = "head_doc_type.npz"

    if env_dir := os.environ.get("PUBGUARD_MODELS_DIR"):
        path = Path(env_dir)
        if path.exists():
            return path

    # Package data
    pkg = Path(__file__).parent / "models"
    if (pkg / marker).exists():
        return pkg

    # CWD
    cwd = Path.cwd() / "pubguard_models"
    if (cwd / marker).exists():
        return cwd

    # Repo dev path (pub_check/models)
    repo = Path(__file__).parent.parent.parent / "models"
    if (repo / marker).exists():
        return repo

    # User home (default install location)
    home = Path.home() / ".pubguard" / "models"
    if (home / marker).exists():
        return home

    # Fallback — use home dir even if empty (training will populate it)
    home.mkdir(parents=True, exist_ok=True)
    return home


# ── Label schemas ────────────────────────────────────────────────

DOC_TYPE_LABELS: List[str] = [
    "scientific_paper",   # Full research article / journal paper
    "poster",             # Conference poster (often single-page, visual)
    "abstract_only",      # Standalone abstract without full paper body
    "junk",               # Flyers, advertisements, non-scholarly PDFs
]

AI_DETECT_LABELS: List[str] = [
    "human",
    "ai_generated",
]

TOXICITY_LABELS: List[str] = [
    "clean",
    "toxic",
]


@dataclass
class PubGuardConfig:
    """Runtime configuration for PubGuard."""

    # ── Embedding backbone ──────────────────────────────────────
    # Re-use the same distilled model you already cache for OpenAlex
    # to avoid downloading a second 50 MB blob.  Any model2vec-
    # compatible StaticModel works here.
    model_name: str = "minishlab/potion-base-32M"
    embedding_dim: int = 512          # potion-base-32M output dim

    # ── Per-head thresholds ─────────────────────────────────────
    # These are posterior-probability thresholds from the softmax
    # head; anything below is "uncertain" and falls back to the
    # majority class.  Calibrate on held-out data.
    doc_type_threshold: float = 0.50
    ai_detect_threshold: float = 0.55
    toxicity_threshold: float = 0.50

    # ── Pipeline gate logic ─────────────────────────────────────
    # The overall `.screen()` returns pass=True ONLY when doc_type
    # is 'scientific_paper'.  Posters, abstracts, and junk are all
    # blocked — the PubVerse pipeline processes publications only.
    # AI detection and toxicity are informational by default.
    require_scientific: bool = True
    block_ai_generated: bool = False   # informational by default
    block_toxic: bool = False          # informational by default

    # ── Batch / performance ─────────────────────────────────────
    batch_size: int = 256
    max_text_chars: int = 4000  # Truncate long texts for embedding

    # ── Paths ───────────────────────────────────────────────────
    models_dir: Optional[Path] = None

    def __post_init__(self):
        if self.models_dir is None:
            self.models_dir = _find_models_dir()
        self.models_dir = Path(self.models_dir)

    # Derived paths
    @property
    def distilled_model_path(self) -> Path:
        return self.models_dir / "pubguard-embedding"

    @property
    def doc_type_head_path(self) -> Path:
        return self.models_dir / "head_doc_type.npz"

    @property
    def ai_detect_head_path(self) -> Path:
        return self.models_dir / "head_ai_detect.npz"

    @property
    def toxicity_head_path(self) -> Path:
        return self.models_dir / "head_toxicity.npz"

    @property
    def label_schemas(self) -> Dict[str, List[str]]:
        return {
            "doc_type": DOC_TYPE_LABELS,
            "ai_detect": AI_DETECT_LABELS,
            "toxicity": TOXICITY_LABELS,
        }

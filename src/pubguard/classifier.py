"""
PubGuard — Multi-head Publication Gatekeeper
=============================================

Architecture
~~~~~~~~~~~~

    ┌─────────────┐
    │  PDF text    │
    └──────┬──────┘
           │
    ┌──────▼──────┐     ┌───────────────────┐
    │  clean_text │────►│  model2vec encode  │──► emb ∈ R^512
    └─────────────┘     └───────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
    │ doc_type head    │ │ ai_detect    │ │ toxicity     │
    │ (concat struct)  │ │ head         │ │ head         │
    │ W·[emb;feat]+b   │ │ W·emb + b    │ │ W·emb + b    │
    │ → softmax(4)     │ │ → softmax(2) │ │ → softmax(2) │
    └─────────────────┘ └──────────────┘ └──────────────┘

Each head is a single linear layer stored as a numpy .npz file
(weights W and bias b).  Inference is pure numpy — no torch needed
at prediction time, matching the openalex classifier's deployment
philosophy.

The doc_type head additionally receives 14 structural features
(section headings present, citation density, etc.) concatenated
with the embedding — these are powerful priors that cost ~0 compute.

Performance target: ≥2,000 records/sec on CPU (same ballpark as
openalex classifier at ~3,000/sec).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import PubGuardConfig, DOC_TYPE_LABELS, AI_DETECT_LABELS, TOXICITY_LABELS
from .text import clean_text, extract_structural_features, STRUCTURAL_FEATURE_NAMES, N_STRUCTURAL_FEATURES

logger = logging.getLogger(__name__)


class LinearHead:
    """
    Single linear classifier head: logits = X @ W + b → softmax.

    Stored as .npz with keys 'W', 'b', 'labels'.
    """

    def __init__(self, labels: List[str]):
        self.labels = labels
        self.n_classes = len(labels)
        self.W: Optional[np.ndarray] = None  # (input_dim, n_classes)
        self.b: Optional[np.ndarray] = None  # (n_classes,)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=True)
        self.W = data["W"]
        self.b = data["b"]
        stored_labels = data.get("labels", None)
        if stored_labels is not None:
            self.labels = list(stored_labels)
            self.n_classes = len(self.labels)
        return True

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, labels=np.array(self.labels))

    def predict(self, X: np.ndarray) -> tuple:
        """
        Returns (pred_labels, pred_scores) for batch.

        X : (batch, input_dim)
        """
        logits = X @ self.W + self.b                 # (batch, n_classes)
        probs = _softmax(logits)                     # (batch, n_classes)
        pred_idx = np.argmax(probs, axis=1)          # (batch,)
        pred_scores = probs[np.arange(len(X)), pred_idx]
        pred_labels = [self.labels[i] for i in pred_idx]
        return pred_labels, pred_scores, probs


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class PubGuard:
    """
    Multi-head publication screening classifier.

    Usage:
        guard = PubGuard()
        guard.initialize()

        # Single document
        verdict = guard.screen("Introduction: We present a novel ...")

        # Batch
        verdicts = guard.screen_batch(["text1", "text2", ...])
    """

    def __init__(self, config: Optional[PubGuardConfig] = None):
        self.config = config or PubGuardConfig()
        self.model = None
        self.head_doc_type = LinearHead(DOC_TYPE_LABELS)
        self.head_ai_detect = LinearHead(AI_DETECT_LABELS)
        self.head_toxicity = LinearHead(TOXICITY_LABELS)
        self._initialized = False

    # ── Initialisation ──────────────────────────────────────────

    def initialize(self) -> bool:
        """Load embedding model + all classification heads."""
        if self._initialized:
            return True

        logger.info("Initializing PubGuard...")
        start = time.time()

        self._load_model()
        self._load_heads()

        self._initialized = True
        logger.info(f"PubGuard initialized in {time.time()-start:.1f}s")
        return True

    def _load_model(self):
        """Load model2vec StaticModel (same as openalex classifier)."""
        from model2vec import StaticModel

        cache = self.config.distilled_model_path
        if cache.exists():
            logger.info(f"Loading embedding model from {cache}")
            self.model = StaticModel.from_pretrained(str(cache))
        else:
            logger.info(f"Downloading model: {self.config.model_name}")
            self.model = StaticModel.from_pretrained(self.config.model_name)
            cache.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(cache))
            logger.info(f"Cached to {cache}")

    def _load_heads(self):
        """Load each classification head from .npz files."""
        heads = [
            ("doc_type",   self.head_doc_type,   self.config.doc_type_head_path),
            ("ai_detect",  self.head_ai_detect,  self.config.ai_detect_head_path),
            ("toxicity",   self.head_toxicity,    self.config.toxicity_head_path),
        ]
        for name, head, path in heads:
            if head.load(path):
                logger.info(f"  Loaded {name} head: {path}")
            else:
                logger.warning(
                    f"  {name} head not found at {path} — "
                    f"run `python -m pubguard.train` first"
                )

    # ── Inference ───────────────────────────────────────────────

    def screen(self, text: str) -> Dict[str, Any]:
        """Screen a single document. Returns verdict dict."""
        return self.screen_batch([text])[0]

    def screen_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Screen a batch of documents.

        Returns list of verdict dicts, each containing:
            doc_type:     {label, score}
            ai_generated: {label, score}
            toxicity:     {label, score}
            pass:         bool (overall gate decision)
        """
        if not self._initialized:
            self.initialize()

        if not texts:
            return []

        cfg = self.config

        # ── Preprocess ──────────────────────────────────────────
        cleaned = [clean_text(t, cfg.max_text_chars) for t in texts]

        # ── Embed ───────────────────────────────────────────────
        embeddings = self.model.encode(cleaned)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div-by-zero
        embeddings = (embeddings / norms).astype("float32")

        # ── Structural features (for doc_type head) ─────────────
        struct_feats = np.array(
            [list(extract_structural_features(t).values()) for t in cleaned],
            dtype="float32",
        )
        doc_type_input = np.concatenate([embeddings, struct_feats], axis=1)

        # ── Per-head predictions ────────────────────────────────
        results = []

        has_doc = self.head_doc_type.W is not None
        has_ai  = self.head_ai_detect.W is not None
        has_tox = self.head_toxicity.W is not None

        dt_labels, dt_scores, _ = (
            self.head_doc_type.predict(doc_type_input) if has_doc
            else (["unknown"] * len(texts), [0.0] * len(texts), None)
        )
        ai_labels, ai_scores, _ = (
            self.head_ai_detect.predict(embeddings) if has_ai
            else (["unknown"] * len(texts), [0.0] * len(texts), None)
        )
        tx_labels, tx_scores, _ = (
            self.head_toxicity.predict(embeddings) if has_tox
            else (["unknown"] * len(texts), [0.0] * len(texts), None)
        )

        for i in range(len(texts)):
            # Gate logic
            passes = True
            if cfg.require_scientific and dt_labels[i] not in ("scientific_paper", "poster"):
                passes = False
            if cfg.block_ai_generated and ai_labels[i] == "ai_generated":
                passes = False
            if cfg.block_toxic and tx_labels[i] == "toxic":
                passes = False

            results.append({
                "doc_type": {
                    "label": dt_labels[i],
                    "score": round(float(dt_scores[i]), 4),
                },
                "ai_generated": {
                    "label": ai_labels[i],
                    "score": round(float(ai_scores[i]), 4),
                },
                "toxicity": {
                    "label": tx_labels[i],
                    "score": round(float(tx_scores[i]), 4),
                },
                "pass": passes,
            })

        return results

    # ── File-level convenience ──────────────────────────────────

    def screen_file(self, path: Path) -> Dict[str, Any]:
        """Read a text file and screen it."""
        text = Path(path).read_text(errors="replace")
        return self.screen(text)

"""
Training pipeline for PubGuard classification heads.

Trains lightweight linear classifiers on frozen model2vec embeddings.
This follows the same paradigm as the openalex-topic-classifier:
the expensive embedding is pre-computed once, and the classifier
itself is a single matrix multiply — fast to train, fast to infer.

Training strategy:
    1. Load + cache model2vec embeddings for all training data
    2. For each head, fit a logistic regression (sklearn) with
       class-balanced weights and L2 regularisation
    3. Export weights as .npz for the numpy-only inference path
    4. Report per-class precision / recall / F1 on held-out split

The entire pipeline trains in <5 minutes on CPU for ~50K samples,
consistent with your existing toolchain.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .config import PubGuardConfig, DOC_TYPE_LABELS, AI_DETECT_LABELS, TOXICITY_LABELS
from .classifier import LinearHead
from .text import clean_text, extract_structural_features, N_STRUCTURAL_FEATURES

logger = logging.getLogger(__name__)


def load_ndjson(path: Path) -> Tuple[List[str], List[str]]:
    """Load NDJSON file → (texts, labels)."""
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                labels.append(row["label"])
    return texts, labels


def embed_texts(
    texts: List[str],
    config: PubGuardConfig,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Encode texts with model2vec, L2-normalise, return (N, D) float32.

    Optionally caches to disk to avoid re-embedding on repeat runs.
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    from model2vec import StaticModel

    model_path = config.distilled_model_path
    if model_path.exists():
        model = StaticModel.from_pretrained(str(model_path))
    else:
        model = StaticModel.from_pretrained(config.model_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_path))

    logger.info(f"Embedding {len(texts)} texts...")
    cleaned = [clean_text(t, config.max_text_chars) for t in texts]
    embeddings = model.encode(cleaned, show_progress_bar=True)

    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = (embeddings / norms).astype("float32")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Cached embeddings to {cache_path}")

    return embeddings


def compute_structural_features(texts: List[str]) -> np.ndarray:
    """Compute structural features for all texts."""
    feats = []
    for t in texts:
        cleaned = clean_text(t)
        feat_dict = extract_structural_features(cleaned)
        feats.append(list(feat_dict.values()))
    return np.array(feats, dtype="float32")


def train_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    head_name: str,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LinearHead:
    """
    Train a single linear classification head.

    Uses sklearn LogisticRegression with:
        - L2 regularisation (C parameter)
        - class_weight='balanced' for imbalanced data
        - lbfgs solver (good for moderate feature counts)
        - multinomial objective even for binary (consistent API)

    Extracts W and b into a LinearHead for numpy-only inference.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {head_name} head")
    logger.info(f"{'='*60}")
    logger.info(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    logger.info(f"  Features: {X_train.shape[1]} | Classes: {len(labels)}")

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"    {u}: {c:,}")

    start = time.time()

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    elapsed = time.time() - start
    logger.info(f"  Trained in {elapsed:.1f}s")

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    logger.info(f"\n{report}")

    # Extract weights into LinearHead
    head = LinearHead(labels)
    # sklearn stores coef_ as (n_classes, n_features) for multinomial
    # We want W as (n_features, n_classes) for X @ W + b
    if clf.coef_.shape[0] == 1:
        # Binary case: sklearn only stores one row
        # Expand to full 2-class format
        head.W = np.vstack([-clf.coef_[0], clf.coef_[0]]).T.astype("float32")
        head.b = np.array([-clf.intercept_[0], clf.intercept_[0]], dtype="float32")
    else:
        head.W = clf.coef_.T.astype("float32")  # (features, classes)
        head.b = clf.intercept_.astype("float32")

    # Sanity check: reproduce sklearn predictions
    logits = X_test[:5] @ head.W + head.b
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    np_pred_idx = np.argmax(probs, axis=1)
    sk_pred_idx = clf.predict(X_test[:5])  # returns integer class indices
    assert list(np_pred_idx) == list(int(x) for x in sk_pred_idx), \
        f"Mismatch: {list(np_pred_idx)} vs {list(sk_pred_idx)}"
    logger.info("  ✓ Numpy inference matches sklearn predictions")

    return head


def train_all(
    data_dir: Path,
    config: Optional[PubGuardConfig] = None,
    test_size: float = 0.15,
):
    """
    Train all three classification heads.

    Args:
        data_dir: Directory containing the prepared NDJSON files
        config: PubGuard configuration
        test_size: Fraction of data held out for evaluation
    """
    config = config or PubGuardConfig()
    data_dir = Path(data_dir)
    cache_dir = data_dir / "embeddings_cache"

    logger.info("=" * 60)
    logger.info("PubGuard Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Models dir: {config.models_dir}")
    start_total = time.time()

    # ── HEAD 1: doc_type ────────────────────────────────────────
    doc_type_path = data_dir / "doc_type_train.ndjson"
    if doc_type_path.exists():
        texts, labels = load_ndjson(doc_type_path)
        label_to_idx = {l: i for i, l in enumerate(DOC_TYPE_LABELS)}

        # Embed
        embeddings = embed_texts(
            texts, config,
            cache_path=cache_dir / "doc_type_emb.npy",
        )

        # Add structural features
        logger.info("Computing structural features...")
        struct = compute_structural_features(texts)
        X = np.concatenate([embeddings, struct], axis=1)

        y = np.array([label_to_idx.get(l, 0) for l in labels])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        head = train_head(X_tr, y_tr, X_te, y_te, DOC_TYPE_LABELS, "doc_type")
        head.save(config.doc_type_head_path)
        logger.info(f"Saved → {config.doc_type_head_path}")
    else:
        logger.warning(f"doc_type data not found: {doc_type_path}")

    # ── HEAD 2: ai_detect ───────────────────────────────────────
    ai_path = data_dir / "ai_detect_train.ndjson"
    if ai_path.exists():
        texts, labels = load_ndjson(ai_path)
        label_to_idx = {l: i for i, l in enumerate(AI_DETECT_LABELS)}

        embeddings = embed_texts(
            texts, config,
            cache_path=cache_dir / "ai_detect_emb.npy",
        )

        y = np.array([label_to_idx.get(l, 0) for l in labels])

        X_tr, X_te, y_tr, y_te = train_test_split(
            embeddings, y, test_size=test_size, stratify=y, random_state=42
        )

        head = train_head(X_tr, y_tr, X_te, y_te, AI_DETECT_LABELS, "ai_detect")
        head.save(config.ai_detect_head_path)
        logger.info(f"Saved → {config.ai_detect_head_path}")
    else:
        logger.warning(f"ai_detect data not found: {ai_path}")

    # ── HEAD 3: toxicity ────────────────────────────────────────
    tox_path = data_dir / "toxicity_train.ndjson"
    if tox_path.exists():
        texts, labels = load_ndjson(tox_path)
        label_to_idx = {l: i for i, l in enumerate(TOXICITY_LABELS)}

        embeddings = embed_texts(
            texts, config,
            cache_path=cache_dir / "toxicity_emb.npy",
        )

        y = np.array([label_to_idx.get(l, 0) for l in labels])

        X_tr, X_te, y_tr, y_te = train_test_split(
            embeddings, y, test_size=test_size, stratify=y, random_state=42
        )

        head = train_head(X_tr, y_tr, X_te, y_te, TOXICITY_LABELS, "toxicity")
        head.save(config.toxicity_head_path)
        logger.info(f"Saved → {config.toxicity_head_path}")
    else:
        logger.warning(f"toxicity data not found: {tox_path}")

    elapsed = time.time() - start_total
    logger.info(f"\nTotal training time: {elapsed/60:.1f} minutes")
    logger.info("All heads saved to: " + str(config.models_dir))

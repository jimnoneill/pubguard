"""
PubVerse Error Code System
==========================

Structured error codes for the entire PubVerse pipeline.
PubGuard codes (PV-0XXX) encode classifier predictions directly
into the code digits.

Error code format:  PV-SXNN
    S  = Step number (0-8)
    X  = Sub-category
    NN = Detail

PubGuard composite encoding (Step 0):
    PV-0 [doc_type] [ai_detect] [toxicity]
         0=paper     0=human     0=clean
         1=poster    1=ai        1=toxic
         2=abstract
         3=junk
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

# ── PubGuard (Step 0) error messages ─────────────────────────────

# Snarky messages keyed by doc_type classification
DOC_TYPE_MESSAGES = {
    "scientific_paper": "Welcome to the lab.",
    "poster": (
        "That's a poster, not a paper. We appreciate the aesthetic effort, "
        "but we need Methods, not bullet points on a corkboard."
    ),
    "abstract_only": (
        "We got the trailer but not the movie. "
        "Where's the rest of the paper?"
    ),
    "junk": (
        "That's not a paper, that's a cry for help. Pool party invitations, "
        "invoices, and fantasy football drafts do not constitute peer-reviewed research."
    ),
}

AI_DETECT_MESSAGES = {
    "human": None,  # No message needed
    "ai_generated": (
        "Our classifier thinks a robot wrote this. "
        "The Turing test starts at the Introduction."
    ),
}

TOXICITY_MESSAGES = {
    "clean": None,
    "toxic": (
        "Content flagged as potentially toxic. "
        "Science should be provocative, not offensive."
    ),
}

# Special composite messages for particularly entertaining combos
COMBO_MESSAGES = {
    (3, 1, 0): "AI-generated junk. Congratulations, you've automated mediocrity.",
    (3, 0, 1): "Toxic junk. This is somehow worse than a pool party flyer.",
    (3, 1, 1): "The trifecta. AI-generated toxic junk. We'd be impressed if we weren't horrified.",
    (1, 1, 0): "An AI-generated poster. The future is here and it's making conference posters.",
    (2, 1, 0): "An AI-generated abstract with no paper attached. Peak efficiency.",
}

# Class label → index mapping (matches config.py label order)
DOC_TYPE_INDEX = {"scientific_paper": 0, "poster": 1, "abstract_only": 2, "junk": 3}
AI_DETECT_INDEX = {"human": 0, "ai_generated": 1}
TOXICITY_INDEX = {"clean": 0, "toxic": 1}


@dataclass
class PubVerseError:
    """Structured pipeline error code."""
    code: str           # e.g. "PV-0300"
    name: str           # e.g. "JUNK_DETECTED"
    message: str        # Human-readable (snarky) description
    step: int           # Pipeline step number
    fatal: bool         # Whether this should halt the pipeline
    details: Optional[Dict[str, Any]] = None  # Optional scores, labels, etc.

    def __str__(self) -> str:
        return f"{self.code} | {self.name} | {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "code": self.code,
            "name": self.name,
            "message": self.message,
            "step": self.step,
            "fatal": self.fatal,
        }
        if self.details:
            d["details"] = self.details
        return d


def build_pubguard_error(verdict: Dict[str, Any]) -> PubVerseError:
    """
    Build a PubGuard error code from a screening verdict.

    The code encodes the classifier predictions:
        PV-0[doc_type_idx][ai_detect_idx][toxicity_idx]

    Returns PV-0000 (ALL_CLEAR) if the paper passes.
    """
    dt_label = verdict["doc_type"]["label"]
    ai_label = verdict["ai_generated"]["label"]
    tx_label = verdict["toxicity"]["label"]

    dt_idx = DOC_TYPE_INDEX.get(dt_label, 9)
    ai_idx = AI_DETECT_INDEX.get(ai_label, 9)
    tx_idx = TOXICITY_INDEX.get(tx_label, 9)

    code = f"PV-0{dt_idx}{ai_idx}{tx_idx}"

    # Build name
    if verdict["pass"]:
        name = "ALL_CLEAR"
    else:
        parts = []
        if dt_idx > 0:
            parts.append(dt_label.upper())
        if ai_idx > 0:
            parts.append("AI_GENERATED")
        if tx_idx > 0:
            parts.append("TOXIC")
        name = "_AND_".join(parts) if parts else "REJECTED"

    # Build message — check combo messages first, then individual
    combo_key = (dt_idx, ai_idx, tx_idx)
    if combo_key in COMBO_MESSAGES:
        message = COMBO_MESSAGES[combo_key]
    elif dt_idx > 0:
        message = DOC_TYPE_MESSAGES.get(dt_label, "Unknown document type.")
    elif ai_idx > 0:
        message = AI_DETECT_MESSAGES.get(ai_label, "AI content detected.")
    elif tx_idx > 0:
        message = TOXICITY_MESSAGES.get(tx_label, "Toxic content detected.")
    else:
        message = "Welcome to the lab."

    # Add scores to message
    score_parts = []
    if dt_idx > 0:
        score_parts.append(f"doc_type={dt_label}:{verdict['doc_type']['score']:.3f}")
    if ai_idx > 0:
        score_parts.append(f"ai={verdict['ai_generated']['score']:.3f}")
    if tx_idx > 0:
        score_parts.append(f"toxicity={verdict['toxicity']['score']:.3f}")

    if score_parts:
        message += f" ({', '.join(score_parts)})"

    # Fatal = doc_type is not scientific_paper (hard gate)
    fatal = dt_idx > 0

    details = {
        "doc_type": verdict["doc_type"],
        "ai_generated": verdict["ai_generated"],
        "toxicity": verdict["toxicity"],
    }

    return PubVerseError(
        code=code,
        name=name,
        message=message,
        step=0,
        fatal=fatal,
        details=details,
    )


# ── Special PubGuard errors ──────────────────────────────────────

def empty_input_error() -> PubVerseError:
    return PubVerseError(
        code="PV-0900",
        name="EMPTY_INPUT",
        message=(
            "You sent us nothing. Literally nothing. "
            "The void does not require peer review."
        ),
        step=0,
        fatal=True,
    )


def unreadable_pdf_error(filename: str = "") -> PubVerseError:
    return PubVerseError(
        code="PV-0901",
        name="UNREADABLE_PDF",
        message=(
            f"We can't read this PDF{f' ({filename})' if filename else ''}. "
            "If your PDF parser can't parse it, maybe it's not a PDF."
        ),
        step=0,
        fatal=True,
    )


def models_missing_error() -> PubVerseError:
    return PubVerseError(
        code="PV-0902",
        name="MODELS_MISSING",
        message=(
            "PubGuard models not found. "
            "Run: cd pub_check && python scripts/train_pubguard.py"
        ),
        step=0,
        fatal=False,  # Pipeline can continue without PubGuard
    )


def gate_bypassed() -> PubVerseError:
    return PubVerseError(
        code="PV-0999",
        name="GATE_BYPASSED",
        message="PubGuard screening bypassed (PUBGUARD_STRICT=0). Proceeding on faith. Good luck.",
        step=0,
        fatal=False,
    )


# ── Pipeline step errors (Steps 1-8) ────────────────────────────

def pipeline_error(step: int, sub: int, detail: int,
                   name: str, message: str, fatal: bool = True) -> PubVerseError:
    """Create a pipeline error for steps 1-8."""
    code = f"PV-{step}{sub}{detail:02d}"
    return PubVerseError(code=code, name=name, message=message, step=step, fatal=fatal)


# Pre-built pipeline errors for bash scripts to reference by name
PIPELINE_ERRORS = {
    # Step 1 — Feature Extraction
    "PV-1100": ("EXTRACTION_FAILED", "VLM feature extraction failed. Your PDF defeated a 7-billion parameter model. Impressive, actually.", True),
    "PV-1101": ("NO_TSV_OUTPUT", "Extraction ran but produced no output file. The VLM started reading and apparently gave up.", True),
    "PV-1103": ("GPU_DRIVER_ERROR", "nvidia-smi reports a driver problem. The VLM will likely hang forever waiting for a GPU that isn't home.", True),
    "PV-1200": ("VLM_NOT_FOUND", "Feature extraction script not found. Did someone move it?", True),
    # Step 2 — PubVerse Analysis
    "PV-2100": ("ANALYSIS_FAILED", "PubVerse analysis crashed. This is the big one — check the logs.", True),
    # Step 3 — Artifact Verification
    "PV-3100": ("MATRIX_MISSING", "Unified adjacency matrix not found. Something went sideways in clustering.", True),
    "PV-3101": ("PICKLE_MISSING", "Impact analysis pickle not found. The most important intermediate file is AWOL.", True),
    # Step 4 — Graph Construction
    "PV-4100": ("GRAPH_FAILED", "Graph construction failed. The nodes existed briefly, like a postdoc's optimism.", True),
    # Step 5 — 42DeepThought Scoring
    "PV-5100": ("SCORING_FAILED", "GNN scoring crashed. Check CUDA, check the graph, check your assumptions.", True),
    "PV-5101": ("NO_GRAPH_PICKLE", "Graph pickle not found for scoring. Step 4 must have failed silently.", True),
    "PV-5102": ("NO_SCORES_OUTPUT", "Scoring ran but produced no TSV. The GNN had nothing to say about your paper.", False),
    "PV-5200": ("DEEPTHOUGHT_MISSING", "42DeepThought directory not found. The entire scoring engine is missing.", False),
    # Step 6 — Cluster Analysis
    "PV-6100": ("CLUSTER_FAILED", "Cluster analysis crashed. Your paper is a loner — or the code is.", False),
    "PV-6102": ("DB_TIMEOUT", "Cluster database population timed out (>1 hour). The LLM is still thinking.", False),
    "PV-6103": ("NO_QUERY_ID", "Could not extract query paper ID from TSV. Who are you, even?", True),
    # Step 7 — Enrichment
    "PV-7100": ("ENRICH_FAILED", "Enrichment script crashed. The data refused to be unified.", False),
    # Step 8 — Visualization
    "PV-8100": ("VIZ_FAILED", "Visualization generation failed. You'll have to imagine the graph.", False),
}


def format_error_line(code: str, name: str = None, message: str = None) -> str:
    """Format a single error line for stdout output."""
    if name is None or message is None:
        if code in PIPELINE_ERRORS:
            name, message, _ = PIPELINE_ERRORS[code]
        else:
            name = name or "UNKNOWN"
            message = message or "An error occurred."
    return f"{code} | {name} | {message}"

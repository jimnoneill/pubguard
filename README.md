<div align="center">
  <img src="PubGuard.png" alt="PubGuard Logo" width="400"/>
</div>

# PubGuard — Multi-Head Scientific Publication Gatekeeper

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/jimnoneill/pubguard-classifier)
[![HuggingFace Data](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/jimnoneill/pubguard-training-data)

PubGuard is a lightweight, CPU-optimized document classifier that screens PDF text to determine whether it represents a genuine scientific publication. It rejects junk (flyers, invoices, non-scholarly PDFs) before expensive downstream processing.

Designed as **Step 0** in the [PubVerse + 42DeepThought](https://github.com/jimnoneill/pubverse_42dt) pipeline — runs in 3.3ms per document, no GPU needed.

## Three Classification Heads

| Head | Classes | Accuracy | What it detects |
|------|---------|----------|-----------------|
| **doc_type** | 4 | **99.7%** | scientific_paper · poster · abstract_only · junk |
| **ai_detect** | 2 | 83.4% | human · ai_generated |
| **toxicity** | 2 | 84.7% | clean · toxic |

Each head is a single linear layer stored as a `.npz` file (8–12 KB). Inference is pure numpy — no torch needed.

## Installation

```bash
pip install git+https://github.com/jimnoneill/pubguard.git
```

With training dependencies:

```bash
pip install "pubguard[train] @ git+https://github.com/jimnoneill/pubguard.git"
```

Or install locally for development:

```bash
git clone https://github.com/jimnoneill/pubguard.git
cd pubguard
pip install -e ".[train]"
```

## Quick Start

### Screen a document

```python
from pubguard import PubGuard

guard = PubGuard()
guard.initialize()

verdict = guard.screen("Introduction: We present a novel deep learning approach...")
print(verdict)
# {
#   'doc_type': {'label': 'scientific_paper', 'score': 0.994},
#   'ai_generated': {'label': 'human', 'score': 0.875},
#   'toxicity': {'label': 'clean', 'score': 0.999},
#   'pass': True
# }
```

### Screen a PDF file

```python
import fitz  # PyMuPDF

doc = fitz.open("paper.pdf")
text = " ".join(page.get_text() for page in doc)
doc.close()

verdict = guard.screen(text[:8000])
if verdict["pass"]:
    print("Valid scientific publication — proceed with analysis")
else:
    print(f"Rejected: {verdict['doc_type']['label']}")
```

### Batch screening

```python
verdicts = guard.screen_batch(["text1", "text2", "text3"])
```

## Gate Logic

Both `scientific_paper` and `poster` pass the gate — they're valid scientific content. Only `abstract_only` and `junk` are blocked:

```
scientific_paper  →  ✅ PASS
poster            →  ✅ PASS
abstract_only     →  ❌ BLOCKED
junk              →  ❌ BLOCKED
```

AI detection and toxicity are **informational by default** — reported but not blocking.

## Pipeline Integration

Drop into any bash pipeline:

```bash
# Extract text from PDF
PDF_TEXT=$(python3 -c "import fitz; d=fitz.open('$PDF'); print(' '.join(p.get_text() for p in d)[:8000])")

# Screen it
echo "$PDF_TEXT" | python3 scripts/pubguard_gate.py 2>/dev/null
if [ $? -ne 0 ]; then
    echo "REJECTED — not a valid scientific publication"
    exit 1
fi
```

## Training

### Install training dependencies

```bash
pip install -e ".[train]"
```

### Train all three heads

```bash
python scripts/train_pubguard.py --data-dir ./pubguard_data --n-per-class 15000
```

Downloads datasets from HuggingFace, embeds with model2vec, trains sklearn LogisticRegression heads. Completes in ~1 minute on CPU.

### Training Data Sources

| Head | Sources |
|------|---------|
| **doc_type** | [armanc/scientific_papers](https://huggingface.co/datasets/armanc/scientific_papers), [gfissore/arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021), [ag_news](https://huggingface.co/datasets/ag_news), [poster-sentry-training-data](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data) |
| **ai_detect** | [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid), [NicolaiSivesind/ChatGPT-Research-Abstracts](https://huggingface.co/datasets/NicolaiSivesind/ChatGPT-Research-Abstracts) |
| **toxicity** | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments), [skg/toxigen-data](https://huggingface.co/datasets/skg/toxigen-data) |

The poster class uses real scientific poster text from the [posters.science](https://posters.science) corpus via [PosterSentry](https://huggingface.co/fairdataihub/poster-sentry).

## Architecture

```
┌─────────────┐
│  PDF text    │
└──────┬──────┘
       │
  model2vec encode  ──► emb ∈ R^512
       │
       ├─────────────────┬─────────────────┐
       ▼                 ▼                 ▼
 ┌───────────┐    ┌───────────┐    ┌───────────┐
 │ doc_type  │    │ ai_detect │    │ toxicity  │
 │ [emb+feat]│    │ [emb]     │    │ [emb]     │
 │ →softmax4 │    │ →softmax2 │    │ →softmax2 │
 └───────────┘    └───────────┘    └───────────┘
```

Same embedding backbone as the [OpenAlex Topic Classifier](https://github.com/jimnoneill/openalex-topic-classifier) — shares the cached model2vec weights.

## Project Structure

```
pubguard/
├── src/pubguard/
│   ├── __init__.py          # PubGuard, PubGuardConfig exports
│   ├── classifier.py        # PubGuard class — screen(), screen_batch()
│   ├── config.py            # Configuration + model path resolution
│   ├── text.py              # Text cleaning + structural feature extraction
│   ├── train.py             # Training pipeline (sklearn LogisticRegression)
│   ├── data.py              # Dataset download + preparation
│   ├── errors.py            # PV-XXXX error code system
│   └── cli.py               # CLI interface
├── scripts/
│   ├── pubguard_gate.py     # Bash pipeline integration (exit 0/1)
│   └── train_pubguard.py    # Training entry point
├── ERRORS.md                # Error code reference guide
├── PubGuard.png             # Logo
└── pyproject.toml           # pip-installable package
```

## HuggingFace

| Resource | Link |
|----------|------|
| **Trained model** | [jimnoneill/pubguard-classifier](https://huggingface.co/jimnoneill/pubguard-classifier) |
| **Training data** | [jimnoneill/pubguard-training-data](https://huggingface.co/datasets/jimnoneill/pubguard-training-data) |

## License

MIT License — See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{pubguard_2026,
  title = {PubGuard: Multi-Head Scientific Publication Gatekeeper},
  author = {O'Neill, James},
  year = {2026},
  url = {https://github.com/jimnoneill/pubguard}
}
```

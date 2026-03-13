# PubVerse Error Codes

> *"Every rejection is a learning opportunity. Mostly for the person who tried to sneak a pool party flyer into a metagenomics pipeline."*

## Error Code Format

```
PV-SXNN
 │ ││└┘
 │ ││ └─ Detail code (00-99)
 │ │└─── Sub-category
 │ └──── Step number (0-8)
 └────── PubVerse prefix
```

All error codes are printed to **stdout** as a single line:

```
PV-0400 | JUNK_DETECTED | That's not a paper, that's a cry for help (doc_type=junk:0.998)
```

---

## Step 0 — PubGuard Screening

The bouncer at the door. If your PDF can't get past here, it definitely
shouldn't be anywhere near a GNN.

The PubGuard codes embed the classifier indices directly:
`PV-0{doc_type}{ai_detect}{toxicity}` where each digit is the
predicted class index.

| Code | Name | What Happened |
|------|------|---------------|
| **PV-0000** | `ALL_CLEAR` | Paper passed screening. Welcome to the lab. |
| **PV-0100** | `LITERATURE_REVIEW` | That's a review article, not original research. We appreciate the bibliography, but we need data, not a guided tour of everyone else's. |
| **PV-0200** | `POSTER_DETECTED` | That's a poster, not a paper. We appreciate the aesthetic effort, but we need Methods, not bullet points on a corkboard. |
| **PV-0300** | `ABSTRACT_ONLY` | We got the trailer but not the movie. Where's the rest of the paper? |
| **PV-0400** | `JUNK_DETECTED` | That's not a paper, that's a cry for help. Pool party invitations, invoices, and fantasy football drafts do not constitute peer-reviewed research. |
| **PV-0010** | `AI_GENERATED` | Our classifier thinks a robot wrote this. Not necessarily disqualifying, but noted for the record. The Turing test starts at the Introduction. |
| **PV-0001** | `TOXIC_CONTENT` | Content flagged as potentially toxic. Science should be provocative, not offensive. |
| **PV-0110** | `REVIEW_AND_AI` | An AI-generated literature review. The robots are now reviewing each other's work. |
| **PV-0210** | `POSTER_AND_AI` | An AI-generated poster. The future is here and it's making conference posters. |
| **PV-0310** | `ABSTRACT_AI` | An AI-generated abstract with no paper attached. Peak efficiency. |
| **PV-0410** | `JUNK_AND_AI` | AI-generated junk. Congratulations, you've automated mediocrity. |
| **PV-0401** | `JUNK_AND_TOXIC` | Toxic junk. This is somehow worse than a pool party flyer. |
| **PV-0411** | `JUNK_AI_TOXIC` | The trifecta. AI-generated toxic junk. We'd be impressed if we weren't horrified. |

### Composite Code Encoding

The three middle digits encode each classifier head's prediction index:

```
PV-0 [doc_type] [ai_detect] [toxicity] NN
      │          │           │
      │          │           └─ 0=clean, 1=toxic
      │          └───────────── 0=human, 1=ai_generated
      └──────────────────────── 0=scientific_paper, 1=literature_review,
                                 2=poster, 3=abstract_only, 4=junk
```

So `PV-0000` = scientific_paper + human + clean = **PASS**.
Any non-zero digit in the doc_type position = **hard gate (blocked)**.
Non-zero in ai/toxicity = **soft flag (reported, not blocked by default)**.

**Gate logic:** Only `scientific_paper` (index 0) passes. Literature reviews,
posters, abstract-only documents, and junk are all blocked. Meta-analyses and
systematic reviews are classified as `scientific_paper` (they pass the gate);
only narrative and scoping reviews are classified as `literature_review`.

### Literature Review Detection

PubGuard blocks narrative and scoping literature reviews because the pipeline
scores *original research contributions* — a review article that surveys
existing work has no novel methods, data, or findings for 42DeepThought to
evaluate. Feeding a review through the GNN would produce misleading scores.

**What gets blocked (PV-0100):**
- Narrative reviews ("A review of recent advances in...")
- Scoping reviews ("Mapping the landscape of...")
- Non-systematic overview articles

**What passes as `scientific_paper`:**
- Systematic reviews (structured methodology, PRISMA, etc.)
- Meta-analyses (quantitative synthesis of prior results)
- These contain original analytical contributions and score meaningfully.

Training data for the `literature_review` class comes from PubMed
PublicationType metadata and OpenAlex review-type articles. The classifier
uses the document's full text structure — not just keywords in the title — to
distinguish reviews from original research.

### Special PubGuard Codes

| Code | Name | Description |
|------|------|-------------|
| **PV-0900** | `EMPTY_INPUT` | You sent us nothing. Literally nothing. The void does not require peer review. |
| **PV-0901** | `UNREADABLE_PDF` | We can't read this PDF. Neither could PyMuPDF. If your PDF parser can't parse it, maybe it's not a PDF. |
| **PV-0902** | `MODELS_MISSING` | PubGuard models not found. Run training first: `cd pub_check && python scripts/train_pubguard.py` |
| **PV-0999** | `GATE_BYPASSED` | PubGuard screening was skipped (PUBGUARD_STRICT=0). Proceeding on faith. Good luck. |

---

## Step 1 — PDF Feature Extraction

Where we turn your lovingly formatted PDF into tab-separated values.
The VLM does its best.

| Code | Name | Description |
|------|------|-------------|
| **PV-1000** | `EXTRACTION_OK` | Features extracted successfully. The VLM read your paper and didn't crash. |
| **PV-1100** | `EXTRACTION_FAILED` | VLM feature extraction failed. Your PDF defeated a 7-billion parameter model. Impressive, actually. |
| **PV-1101** | `NO_TSV_OUTPUT` | Extraction ran but produced no output file. The VLM started reading and apparently gave up. |
| **PV-1102** | `VLM_TIMEOUT` | Feature extraction timed out. Your PDF is either very long or very confusing. |
| **PV-1103** | `GPU_DRIVER_ERROR` | nvidia-smi reports a driver problem. The VLM will likely hang forever waiting for a GPU that isn't home. Fix your drivers first. |
| **PV-1200** | `VLM_NOT_FOUND` | Feature extraction script not found. Did someone move `qwen3_local_feature_extraction_cli.py`? |

---

## Step 2 — PubVerse Analysis

The main event. Clustering, impact analysis, topic modeling — the works.

| Code | Name | Description |
|------|------|-------------|
| **PV-2000** | `ANALYSIS_OK` | PubVerse analysis completed. Your paper has been weighed, measured, and clustered. |
| **PV-2100** | `ANALYSIS_FAILED` | App_v5.py crashed. This is the big one — check the logs. |
| **PV-2101** | `STOPWORDS_MISSING` | Stopwords pickle not found. You can't do NLP without knowing which words to ignore. |
| **PV-2102** | `DATASET2_MISSING` | Reference dataset not found. We need something to compare your paper against. |
| **PV-2103** | `RECYCLE_MISSING` | Recycle overlay pickle not found. The cached cluster state is gone. Time to rebuild (grab coffee). |

---

## Step 3 — Artifact Verification

Making sure Step 2 actually produced what it promised.

| Code | Name | Description |
|------|------|-------------|
| **PV-3000** | `ARTIFACTS_OK` | All expected files found. The pipeline is holding together. |
| **PV-3100** | `MATRIX_MISSING` | Unified adjacency matrix not found. PubVerse ran but didn't produce the matrix. Something went sideways in clustering. |
| **PV-3101** | `PICKLE_MISSING` | Impact analysis pickle not found. The most important intermediate file is AWOL. |

---

## Step 4 — Graph Construction

Building the knowledge graph. Nodes, edges, the whole beautiful mess.

| Code | Name | Description |
|------|------|-------------|
| **PV-4000** | `GRAPH_OK` | Knowledge graph constructed. It's a beautiful web of science. |
| **PV-4100** | `GRAPH_FAILED` | Graph construction failed. 42dt_graph_clean_cli.py didn't make it. |
| **PV-4101** | `GRAPH_PICKLE_MISSING` | Graph pickle not produced. The nodes existed briefly, like a postdoc's optimism. |

---

## Step 5 — 42DeepThought Scoring

The GNN scores your paper. This is where GPUs earn their electricity bill.

| Code | Name | Description |
|------|------|-------------|
| **PV-5000** | `SCORING_OK` | 42DeepThought scored your paper. May the odds be in your favor. |
| **PV-5100** | `SCORING_FAILED` | GNN scoring crashed. Check CUDA, check the graph, check your assumptions. |
| **PV-5101** | `NO_GRAPH_PICKLE` | Graph pickle file not found for scoring. Step 4 must have failed silently. |
| **PV-5102** | `NO_SCORES_OUTPUT` | Scoring ran but produced no TSV. The GNN had nothing to say about your paper. |
| **PV-5103** | `CHECKPOINT_CORRUPT` | Model checkpoint failed to load. Delete `deepthought_model.pt` and retrain. |
| **PV-5200** | `DEEPTHOUGHT_MISSING` | 42DeepThought directory not found. The entire scoring engine is missing. |
| **PV-5201** | `LABELS_MISSING` | `42d_scoring.tsv` not found. No reference labels means no supervised scoring. |

---

## Step 6 — Cluster Similarity Analysis

Finding your paper's neighbors in topic space.

| Code | Name | Description |
|------|------|-------------|
| **PV-6000** | `CLUSTER_OK` | Cluster analysis complete. Your paper's social circle has been mapped. |
| **PV-6100** | `CLUSTER_FAILED` | Cluster analysis crashed. Your paper is a loner — or the code is. |
| **PV-6101** | `NO_SANITY_PICKLE` | Sanitycheck pickle not found. Can't do cluster analysis without clusters. |
| **PV-6102** | `DB_TIMEOUT` | Cluster database population timed out (>1 hour). The LLM is still thinking. |
| **PV-6103** | `NO_QUERY_ID` | Could not extract query paper ID from TSV. Who are you, even? |
| **PV-6200** | `CLUSTER_SKIPPED` | Cluster analysis skipped (prerequisites not met). Not fatal, just lonely. |

---

## Step 7 — Data Enrichment

Merging all the analysis results into one JSON payload.

| Code | Name | Description |
|------|------|-------------|
| **PV-7000** | `ENRICH_OK` | Data enrichment complete. Everything is unified and beautiful. |
| **PV-7100** | `ENRICH_FAILED` | Enrichment script crashed. The data refused to be unified. |
| **PV-7200** | `ENRICH_SKIPPED` | Enrichment skipped (cluster analysis didn't complete). Can't enrich what doesn't exist. |

---

## Step 8 — Interactive Visualization

The grand finale. Turning data into something you can click on.

| Code | Name | Description |
|------|------|-------------|
| **PV-8000** | `VIZ_OK` | Visualization generated. Open the HTML and admire your knowledge graph. |
| **PV-8100** | `VIZ_FAILED` | Visualization generation failed. You'll have to imagine the graph. |
| **PV-8200** | `VIZ_SKIPPED` | Visualization skipped (enrichment didn't complete). No data, no graph, no glory. |

---

## Exit Code Summary

The pipeline exits with the **first fatal error code's step number**:

| Exit Code | Meaning |
|-----------|---------|
| `0` | Success — all steps completed (or non-fatal warnings only) |
| `1` | Step 0 — PubGuard rejected the input |
| `2` | Step 1 — Feature extraction failed |
| `3` | Step 2 — PubVerse analysis failed |
| `4` | Step 3 — Required artifacts missing |
| `5` | Step 4 — Graph construction failed |
| `6` | Step 5 — 42DeepThought scoring failed |
| `7` | Step 6 — Cluster analysis failed (fatal only if no fallback) |
| `8` | Step 7/8 — Enrichment or visualization failed |

---

## Interpreting PubGuard Composite Codes

Quick decoder ring for the `PV-0XYZ` codes:

```
X (doc_type):   0=paper ✅  1=review 📚  2=poster 📋  3=abstract 📄  4=junk 🗑️
Y (ai_detect):  0=human ✍️   1=ai 🤖
Z (toxicity):   0=clean ✅  1=toxic ☠️
```

Examples:
- `PV-0000` → Paper + Human + Clean → **PASS** ✅
- `PV-0100` → Review + Human + Clean → *"That's a review, not original research"* 📚
- `PV-0400` → Junk + Human + Clean → *"That's not a paper"* 🗑️
- `PV-0011` → Paper + AI + Toxic → *"Human-passing but spicy"* ⚠️
- `PV-0411` → Junk + AI + Toxic → *"The absolute worst"* 🚫

---

*PubGuard v0.1.0 — Because science has standards, even if your PDF doesn't.*

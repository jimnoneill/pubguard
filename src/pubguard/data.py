"""
Dataset preparation for PubGuard training.

Downloads publicly available datasets from HuggingFace and assembles
them into the three labelled corpora needed by the training pipeline.

Datasets used (verified available 2026-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Head 1 — Document Type** (scientific_paper | poster | abstract_only | junk)

  Positive (scientific_paper):
    - armanc/scientific_papers (arxiv)  ~300 K full-text articles
      cols: article, abstract, section_names

  Negative (abstract_only):
    - gfissore/arxiv-abstracts-2021     ~2 M abstracts
      cols: abstract (filter length < 600 chars)

  Negative (junk):
    - ag_news (news articles) + synthetic templates (flyers, invoices, etc.)

  Negative (poster):
    - Synthetic poster-style structured text

**Head 2 — AI-Generated Text Detection**

    - liamdugan/raid  – multi-model generations, domain="abstracts"
      cols: model, domain, generation  (model="human" for human text)
    - NicolaiSivesind/ChatGPT-Research-Abstracts – real + GPT-3.5 abstracts
      cols: real_abstract, generated_abstract

**Head 3 — Toxicity**

    - google/civil_comments – 1.8 M comments with toxicity scores (0–1)
      cols: text, toxicity
    - skg/toxigen-data – 274 K annotated toxic/benign statements
      cols: text, toxicity_human (1–5 scale)
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)

# ── Synthetic templates ──────────────────────────────────────────

JUNK_TEMPLATES = [
    "🎉 Annual {event} at {place}! Join us on {date}. Free food and drinks. RSVP to {email}.",
    "FOR SALE: {item}. Great condition. ${price}. Contact {name} at {phone}.",
    "{company} is hiring! We're looking for a {role}. Apply now at {url}.",
    "NOTICE: The {dept} office will be closed on {date} for {reason}. Questions? Call {phone}.",
    "Don't miss our {event}! {date} from {time}. {place}. Tickets: ${price}.",
    "Weekly newsletter from {company}. This week: {topic1}, {topic2}, and more!",
    "Invoice #{num} from {company}. Amount due: ${price}. Payment due by {date}.",
    "Meeting agenda for {date}. 1) {topic1} 2) {topic2} 3) {topic3}. Location: {place}.",
    "URGENT: Your {account} password expires on {date}. Click here to reset: {url}.",
    "Congratulations {name}! You've been selected for our exclusive {event}. Limited spots!",
    "Thank you for your purchase! Order #{num}. Estimated delivery: {date}.",
    "{company} presents the {event}. Keynote by {name}. Register at {url}.",
    "Garage sale this weekend! {place}. {date} {time}. Everything must go!",
    "Happy Birthday to {name} from all of us at {company}! 🎂",
    "POOL PARTY! 🏊 Come join us at {place} on {date}. Bring your swimsuit and sunscreen!",
    "Menu for this week: Monday: {food1}. Tuesday: {food2}. Wednesday: {food3}.",
    "Building maintenance notice: {reason} on {date}. Please plan accordingly.",
    "Lost & Found: {item} found near {place}. Contact front desk to claim.",
    "Fantasy Football League draft is on {date}! Don't forget to submit your picks.",
    "Book club meeting: We're reading '{book}' by {name}. Discussion on {date}.",
    "Hey everyone! Movie night at {place} on {date}. We're watching '{movie}'. Bring popcorn!",
    "Reminder: Staff meeting {date} at {time}. Attendance mandatory. {dept}.",
    "Lost cat! Orange tabby, answers to '{pet_name}'. Last seen near {place}. Call {phone}.",
    "HOT DEAL! {item} only ${price}! Limited time offer. Visit {url}.",
    "Club registration open! Join the {club} club. Meetings every {day} at {time}. {place}.",
    "Fundraiser bake sale! {date} at {place}. All proceeds go to {charity}.",
    "Apartment for rent: 2BR/1BA near {place}. ${price}/month. Pet friendly. Call {phone}.",
    "Yoga class every {day} at {time}. {place}. All levels welcome. Bring your own mat!",
    "IT Alert: System maintenance scheduled for {date}. Expected downtime: {time}. {dept}.",
    "Carpool needed! Driving from {place} to {place2} daily. Contact {name} at {email}.",
]

POSTER_TEMPLATES = [
    "TITLE: {title}\n\nAUTHORS: {authors}\nAFFILIATION: {affil}\n\nINTRODUCTION\n{intro}\n\nMETHODS\n{methods}\n\nRESULTS\n{results}\n\nCONCLUSIONS\n{conclusions}\n\nACKNOWLEDGMENTS\n{ack}",
    "{title}\n{authors} | {affil}\n\nBackground: {intro}\n\nApproach: {methods}\n\nKey Findings:\n• {finding1}\n• {finding2}\n• {finding3}\n\nFuture Work: {future}\n\nContact: {email}",
    "POSTER PRESENTATION\n\n{title}\n\n{authors}\n{affil}\n\nObjective: {intro}\n\nDesign: {methods}\n\nOutcome: {results}\n\nConclusion: {conclusions}",
    "{title}\n\n{authors} ({affil})\n\nAim: {intro}\nMethod: {methods}\nResult: {results}\nSummary: {conclusions}\n\nCorrespondence: {email}",
    "RESEARCH POSTER\n─────────────────────\n{title}\n{authors}\n{affil}\n\n▸ Background\n{intro}\n\n▸ Methods\n{methods}\n\n▸ Results\n• {finding1}\n• {finding2}\n\n▸ Conclusion\n{conclusions}\n\nFunding: {ack}",
]


def _fill_template(template: str) -> str:
    """Fill a template with random plausible values."""
    fillers = {
        "{event}": random.choice(["Pool Party", "BBQ Bash", "Career Fair", "Fundraiser Gala", "Open House", "Trivia Night"]),
        "{place}": random.choice(["Room 201", "Hilton Downtown", "the Community Center", "Central Park", "Building B Courtyard", "Main Auditorium"]),
        "{place2}": random.choice(["Campus North", "Downtown", "Tech Park", "Medical Center"]),
        "{date}": random.choice(["March 15", "June 22", "Sept 5", "November 10", "January 30", "Friday the 13th"]),
        "{email}": "info@example.com",
        "{item}": random.choice(["2019 Honda Civic", "MacBook Pro 16-inch", "Standing Desk", "Mountain Bike", "Vintage Guitar"]),
        "{price}": str(random.randint(10, 5000)),
        "{name}": random.choice(["Dr. Smith", "Jane Doe", "Prof. Chen", "Maria Garcia", "Bob Wilson"]),
        "{phone}": "555-0123",
        "{company}": random.choice(["TechCorp", "BioGen Inc.", "Global Solutions", "Acme Labs", "DataFlow Systems"]),
        "{role}": random.choice(["Data Scientist", "Lab Technician", "Project Manager", "Software Engineer"]),
        "{url}": "https://example.com/apply",
        "{dept}": random.choice(["HR", "Finance", "Engineering", "Admissions", "IT Support"]),
        "{reason}": random.choice(["maintenance", "holiday", "training day", "renovation", "fire drill"]),
        "{time}": random.choice(["2-5 PM", "10 AM - 3 PM", "6-9 PM", "All Day", "Noon"]),
        "{topic1}": random.choice(["Q3 Review", "Budget Update", "New Hires", "Project Status"]),
        "{topic2}": random.choice(["Safety Training", "Holiday Schedule", "IT Migration", "Team Building"]),
        "{topic3}": random.choice(["Parking Changes", "Wellness Program", "Open Q&A"]),
        "{account}": random.choice(["university", "corporate", "cloud storage"]),
        "{num}": str(random.randint(10000, 99999)),
        "{food1}": "Pasta Primavera", "{food2}": "Chicken Tikka", "{food3}": "Fish Tacos",
        "{book}": random.choice(["1984", "Sapiens", "The Gene", "Thinking, Fast and Slow"]),
        "{movie}": random.choice(["Inception", "The Matrix", "Interstellar"]),
        "{pet_name}": random.choice(["Whiskers", "Max", "Luna"]),
        "{club}": random.choice(["Chess", "Photography", "Hiking", "Debate"]),
        "{day}": random.choice(["Monday", "Wednesday", "Friday"]),
        "{charity}": random.choice(["Children's Hospital", "Local Food Bank", "Animal Shelter"]),
        "{title}": random.choice([
            "Effects of Temperature on Enzyme Kinetics in Thermophilic Bacteria",
            "Deep Learning for Medical Image Segmentation: A Systematic Review",
            "Novel Biomarkers in Cardiovascular Disease Progression",
            "Metagenomic Analysis of Coral Reef Microbiomes Under Thermal Stress",
            "CRISPR-Cas9 Editing Efficiency in Human iPSC-Derived Neurons",
        ]),
        "{authors}": random.choice(["A. Smith, B. Jones, C. Lee", "R. Patel, S. Kim, T. Brown", "M. Wang, L. Davis"]),
        "{affil}": random.choice(["University of Example, Dept. of Science", "MIT, CSAIL", "Stanford School of Medicine"]),
        "{intro}": random.choice([
            "Background text about the research problem being investigated.",
            "This study addresses the gap in understanding of X in the context of Y.",
            "Recent advances in Z have highlighted the need for improved W.",
        ]),
        "{methods}": random.choice([
            "We employed a cross-sectional study design with N=200 participants.",
            "Samples were collected from 5 sites and processed using standard protocols.",
            "We developed a convolutional neural network trained on 50K labeled images.",
        ]),
        "{results}": random.choice([
            "Treatment group showed 45% improvement (p<0.01) compared to control.",
            "Our model achieved 94.2% accuracy on the held-out test set.",
            "We identified 23 significantly enriched pathways (FDR < 0.05).",
        ]),
        "{conclusions}": random.choice([
            "Our findings support the hypothesis that X leads to improved Y.",
            "These results demonstrate the feasibility of the proposed approach.",
            "Further validation with larger cohorts is warranted.",
        ]),
        "{finding1}": "Significant reduction in error rate (p<0.001)",
        "{finding2}": "Model outperformed baseline by 15%",
        "{finding3}": "Robust to distribution shift across domains",
        "{future}": "Extend to longitudinal datasets and multi-site validation.",
        "{ack}": random.choice(["Funded by NIH Grant R01-ABC123.", "Supported by NSF Award #1234567."]),
    }
    result = template
    for key, val in fillers.items():
        result = result.replace(key, val)
    return result


def generate_synthetic_junk(n: int = 5000) -> List[Dict[str, str]]:
    """Generate synthetic junk documents."""
    samples = []
    for _ in range(n):
        template = random.choice(JUNK_TEMPLATES)
        text = _fill_template(template)
        samples.append({"text": text, "label": "junk"})
    return samples


def generate_synthetic_posters(n: int = 3000) -> List[Dict[str, str]]:
    """Generate synthetic poster-style documents."""
    samples = []
    for _ in range(n):
        template = random.choice(POSTER_TEMPLATES)
        text = _fill_template(template)
        samples.append({"text": text, "label": "poster"})
    return samples


# ── Head 1: doc_type ────────────────────────────────────────────

def prepare_doc_type_dataset(
    output_dir: Path,
    n_per_class: int = 15000,
) -> Path:
    """
    Assemble and save document-type training data.

    Downloads from HuggingFace and combines with synthetic data.
    Saves as NDJSON: {text, label}
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "doc_type_train.ndjson"
    all_samples = []

    logger.info("=== Preparing doc_type dataset ===")

    # ── scientific_paper ─────────────────────────────────────────
    logger.info("Loading armanc/scientific_papers (arxiv split)...")
    try:
        ds = load_dataset(
            "armanc/scientific_papers", "arxiv",
            split="train", streaming=True, trust_remote_code=True,
        )
        count = 0
        for row in ds:
            if count >= n_per_class:
                break
            # Combine abstract + article body for full-text signal
            abstract = row.get("abstract", "") or ""
            article = row.get("article", "") or ""
            text = (abstract + " " + article)[:4000]
            if len(text.strip()) > 100:
                all_samples.append({"text": text.strip(), "label": "scientific_paper"})
                count += 1
        logger.info(f"  scientific_paper: {count}")
    except Exception as e:
        logger.warning(f"Could not load scientific_papers: {e}")
        # Fallback
        logger.info("Falling back to ccdv/arxiv-summarization...")
        try:
            ds = load_dataset(
                "ccdv/arxiv-summarization",
                split="train", streaming=True, trust_remote_code=True,
            )
            count = 0
            for row in ds:
                if count >= n_per_class:
                    break
                text = ((row.get("abstract", "") or "") + " " + (row.get("article", "") or ""))[:4000]
                if len(text.strip()) > 100:
                    all_samples.append({"text": text.strip(), "label": "scientific_paper"})
                    count += 1
            logger.info(f"  scientific_paper (fallback): {count}")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")

    # ── abstract_only ────────────────────────────────────────────
    logger.info("Loading gfissore/arxiv-abstracts-2021...")
    try:
        ds = load_dataset(
            "gfissore/arxiv-abstracts-2021",
            split="train", streaming=True, trust_remote_code=True,
        )
        count = 0
        for row in ds:
            if count >= n_per_class:
                break
            abstract = row.get("abstract", "")
            if abstract and 50 < len(abstract) < 600:
                all_samples.append({"text": abstract.strip(), "label": "abstract_only"})
                count += 1
        logger.info(f"  abstract_only: {count}")
    except Exception as e:
        logger.warning(f"Could not load arxiv-abstracts: {e}")
        # Fallback: extract abstracts from scientific_papers
        logger.info("Generating abstract_only from scientific_papers abstracts...")
        try:
            ds = load_dataset(
                "armanc/scientific_papers", "arxiv",
                split="train", streaming=True, trust_remote_code=True,
            )
            count = 0
            for row in ds:
                if count >= n_per_class:
                    break
                abstract = row.get("abstract", "")
                if abstract and 50 < len(abstract) < 600:
                    all_samples.append({"text": abstract.strip(), "label": "abstract_only"})
                    count += 1
            logger.info(f"  abstract_only (fallback): {count}")
        except Exception:
            pass

    # ── junk (100% real data — ag_news) ────────────────────────────
    logger.info("Loading ag_news for junk class (full — no synthetic)...")
    try:
        ds = load_dataset(
            "ag_news",
            split="train", streaming=True, trust_remote_code=True,
        )
        count = 0
        for row in ds:
            if count >= n_per_class:
                break
            text = row.get("text", "")
            if len(text) > 30:
                all_samples.append({"text": text.strip(), "label": "junk"})
                count += 1
        logger.info(f"  junk (ag_news): {count}")
    except Exception as e:
        logger.warning(f"Could not load ag_news: {e}")

    # ── poster ────────────────────────────────────────────────────
    # NOTE: Real poster text is nearly identical to paper text in
    # embedding space (both are scientific). PubGuard uses text-only
    # features, so we need SHORT, STRUCTURED poster-style texts that
    # the embedding can distinguish from full papers.
    #
    # Strategy: synthetic poster templates (structured, short) +
    # real poster texts TRUNCATED to first 500 chars (title/authors
    # block, which has distinct formatting from paper introductions).
    logger.info("Loading poster data (structured templates + real poster headers)...")
    poster_count = 0

    # (a) Synthetic templates — provide distinctive poster structure signal
    synth_posters = generate_synthetic_posters(min(n_per_class // 2, 7500))
    all_samples.extend(synth_posters)
    poster_count += len(synth_posters)
    logger.info(f"  poster (synthetic templates): {len(synth_posters)}")

    # (b) Real poster header text (first 500 chars only — title/authors block)
    real_poster_count = 0
    local_poster_data = Path("/home/joneill/pubverse_brett/poster_sentry/poster_texts_for_pubguard.ndjson")
    if not local_poster_data.exists():
        local_poster_data = Path.cwd().parent / "poster_sentry" / "poster_texts_for_pubguard.ndjson"

    if local_poster_data.exists():
        logger.info(f"  Adding real poster headers from: {local_poster_data}")
        with open(local_poster_data) as f:
            for line in f:
                if real_poster_count >= n_per_class // 2:
                    break
                row = json.loads(line)
                if row.get("label") == "poster":
                    # Truncate to header region (title, authors, affiliations)
                    text = row["text"][:500]
                    if len(text) > 50:
                        all_samples.append({"text": text, "label": "poster"})
                        real_poster_count += 1
        poster_count += real_poster_count
        logger.info(f"  poster (real headers, ≤500 chars): {real_poster_count}")
    else:
        # Fill with more synthetic templates if no real data available
        extra = generate_synthetic_posters(n_per_class // 2)
        all_samples.extend(extra)
        poster_count += len(extra)
        logger.info(f"  poster (synthetic fallback): {len(extra)}")

    logger.info(f"  poster total: {poster_count}")

    # ── Shuffle and save ─────────────────────────────────────────
    random.shuffle(all_samples)

    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Report distribution
    dist = {}
    for s in all_samples:
        dist[s["label"]] = dist.get(s["label"], 0) + 1
    logger.info(f"Saved {len(all_samples)} samples to {output_path}")
    for label, count in sorted(dist.items()):
        logger.info(f"  {label}: {count}")

    return output_path


# ── Head 2: ai_detect ───────────────────────────────────────────

def prepare_ai_detect_dataset(
    output_dir: Path,
    n_per_class: int = 20000,
) -> Path:
    """
    Assemble AI-generated text detection training data.

    Sources (all verified available):
        - liamdugan/raid: multi-model generations, domain="abstracts"
          model="human" → human, otherwise → ai_generated
        - NicolaiSivesind/ChatGPT-Research-Abstracts: real + GPT-3.5 abstracts
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ai_detect_train.ndjson"
    human_samples = []
    ai_samples = []

    logger.info("=== Preparing ai_detect dataset ===")

    # ── RAID (scientific abstracts domain) ───────────────────────
    logger.info("Loading liamdugan/raid (abstracts domain)...")
    try:
        ds = load_dataset(
            "liamdugan/raid",
            split="train", streaming=True, trust_remote_code=True,
        )
        human_count = 0
        ai_count = 0
        for row in ds:
            domain = row.get("domain", "")
            if domain != "abstracts":
                continue
            text = row.get("generation", "") or ""
            if not text or len(text) < 50:
                continue
            model = row.get("model", "")
            if model == "human":
                if human_count < n_per_class:
                    human_samples.append({"text": text[:4000], "label": "human"})
                    human_count += 1
            else:
                if ai_count < n_per_class:
                    ai_samples.append({"text": text[:4000], "label": "ai_generated"})
                    ai_count += 1
            if human_count >= n_per_class and ai_count >= n_per_class:
                break
        logger.info(f"  RAID: human={human_count}, ai={ai_count}")
    except Exception as e:
        logger.warning(f"Could not load RAID: {e}")

    # ── ChatGPT-Research-Abstracts ───────────────────────────────
    logger.info("Loading NicolaiSivesind/ChatGPT-Research-Abstracts...")
    try:
        ds = load_dataset(
            "NicolaiSivesind/ChatGPT-Research-Abstracts",
            split="train", streaming=True, trust_remote_code=True,
        )
        h_count = 0
        a_count = 0
        for row in ds:
            real = row.get("real_abstract", "")
            generated = row.get("generated_abstract", "")
            if real and len(real) > 50:
                human_samples.append({"text": real[:4000], "label": "human"})
                h_count += 1
            if generated and len(generated) > 50:
                ai_samples.append({"text": generated[:4000], "label": "ai_generated"})
                a_count += 1
        logger.info(f"  ChatGPT-Abstracts: human={h_count}, ai={a_count}")
    except Exception as e:
        logger.warning(f"Could not load ChatGPT-Research-Abstracts: {e}")

    # ── Balance and save ─────────────────────────────────────────
    min_count = min(len(human_samples), len(ai_samples), n_per_class)
    if min_count == 0:
        logger.error("No AI detection training data available!")
        # Save empty file
        with open(output_path, "w") as f:
            pass
        return output_path

    balanced = (
        random.sample(human_samples, min(min_count, len(human_samples)))
        + random.sample(ai_samples, min(min_count, len(ai_samples)))
    )
    random.shuffle(balanced)

    with open(output_path, "w") as f:
        for sample in balanced:
            f.write(json.dumps(sample) + "\n")

    n_h = sum(1 for s in balanced if s["label"] == "human")
    n_a = sum(1 for s in balanced if s["label"] == "ai_generated")
    logger.info(f"Saved {len(balanced)} samples (human={n_h}, ai={n_a}) to {output_path}")
    return output_path


# ── Head 3: toxicity ────────────────────────────────────────────

def prepare_toxicity_dataset(
    output_dir: Path,
    n_per_class: int = 20000,
) -> Path:
    """
    Assemble toxicity detection training data.

    Sources (all verified available without manual download):
        - google/civil_comments – ~1.8 M comments with toxicity float (0–1)
          We threshold: toxic >= 0.5, clean < 0.1
        - skg/toxigen-data – 274 K annotated statements
          toxicity_human is a float 1–5; we use >= 4.0 as toxic, <= 2.0 as clean
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "toxicity_train.ndjson"
    toxic_samples = []
    clean_samples = []

    logger.info("=== Preparing toxicity dataset ===")

    # ── Civil Comments ───────────────────────────────────────────
    logger.info("Loading google/civil_comments...")
    try:
        ds = load_dataset(
            "google/civil_comments",
            split="train", streaming=True, trust_remote_code=True,
        )
        toxic_count = 0
        clean_count = 0
        for row in ds:
            text = row.get("text", "")
            if not text or len(text) < 20:
                continue
            toxicity = row.get("toxicity", 0.0)
            if toxicity >= 0.5 and toxic_count < n_per_class:
                toxic_samples.append({"text": text[:4000], "label": "toxic"})
                toxic_count += 1
            elif toxicity < 0.1 and clean_count < n_per_class:
                clean_samples.append({"text": text[:4000], "label": "clean"})
                clean_count += 1
            if toxic_count >= n_per_class and clean_count >= n_per_class:
                break
        logger.info(f"  Civil Comments: toxic={toxic_count}, clean={clean_count}")
    except Exception as e:
        logger.warning(f"Could not load civil_comments: {e}")

    # ── ToxiGen ──────────────────────────────────────────────────
    logger.info("Loading skg/toxigen-data...")
    try:
        ds = load_dataset(
            "skg/toxigen-data",
            split="train", streaming=True, trust_remote_code=True,
        )
        t_count = 0
        c_count = 0
        for row in ds:
            text = row.get("text", "")
            if not text or len(text) < 20:
                continue
            # toxicity_human is 1-5 scale
            tox_score = row.get("toxicity_human", None)
            if tox_score is None:
                continue
            tox_score = float(tox_score)
            if tox_score >= 4.0:
                toxic_samples.append({"text": text[:4000], "label": "toxic"})
                t_count += 1
            elif tox_score <= 2.0:
                clean_samples.append({"text": text[:4000], "label": "clean"})
                c_count += 1
        logger.info(f"  ToxiGen: toxic={t_count}, clean={c_count}")
    except Exception as e:
        logger.warning(f"Could not load ToxiGen: {e}")

    # ── Balance and save ─────────────────────────────────────────
    min_count = min(len(toxic_samples), len(clean_samples), n_per_class)
    if min_count == 0:
        logger.error("No toxicity training data available!")
        with open(output_path, "w") as f:
            pass
        return output_path

    balanced = (
        random.sample(toxic_samples, min(min_count, len(toxic_samples)))
        + random.sample(clean_samples, min(min_count, len(clean_samples)))
    )
    random.shuffle(balanced)

    with open(output_path, "w") as f:
        for sample in balanced:
            f.write(json.dumps(sample) + "\n")

    n_t = sum(1 for s in balanced if s["label"] == "toxic")
    n_c = sum(1 for s in balanced if s["label"] == "clean")
    logger.info(f"Saved {len(balanced)} samples (toxic={n_t}, clean={n_c}) to {output_path}")
    return output_path


# ── Orchestrator ─────────────────────────────────────────────────

def prepare_all(output_dir: Path, n_per_class: int = 15000):
    """Download and prepare all three datasets."""
    output_dir = Path(output_dir)
    logger.info(f"Preparing all datasets in {output_dir}")

    paths = {}
    paths["doc_type"] = prepare_doc_type_dataset(output_dir, n_per_class)
    paths["ai_detect"] = prepare_ai_detect_dataset(output_dir, n_per_class)
    paths["toxicity"] = prepare_toxicity_dataset(output_dir, n_per_class)

    logger.info("All datasets prepared!")
    return paths

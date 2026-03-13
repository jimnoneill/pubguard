"""
Dataset preparation for PubGuard training.

Downloads publicly available datasets from HuggingFace and assembles
them into the three labelled corpora needed by the training pipeline.
Optionally ingests a local PDF corpus with PubMed metadata for
real-world scientific_paper / literature_review labels.

Datasets used (verified available 2026-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Head 1 — Document Type** (scientific_paper | literature_review | poster | abstract_only | junk)

  Positive (scientific_paper):
    - Real PDF corpus (microbiome/metagenomics PDFs with PMID filenames)
    - armanc/scientific_papers (arxiv)  ~300 K full-text articles
      cols: article, abstract, section_names

  Negative (literature_review):
    - Real PDF corpus — narrative/scoping reviews identified via PubMed
      PublicationType tags (meta-analyses and systematic reviews pass as
      scientific_paper per user preference)

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
import os
import random
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError

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


# ── PDF corpus helpers ──────────────────────────────────────────


def _extract_pdf_texts(
    corpus_dir: Path,
    max_chars: int = 4000,
    min_chars: int = 100,
) -> Dict[str, str]:
    """
    Walk a PDF corpus directory and extract text from each PDF.

    Returns {pmid: text} mapping. PMIDs are parsed from filenames
    (e.g. "12345678.pdf" → "12345678").
    """
    import fitz  # PyMuPDF

    corpus_dir = Path(corpus_dir)
    results: Dict[str, str] = {}
    failed = 0

    pdf_files = sorted(corpus_dir.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {corpus_dir}")

    for pdf_path in pdf_files:
        pmid = pdf_path.stem  # e.g. "12345678"
        if not pmid.isdigit():
            continue
        try:
            doc = fitz.open(str(pdf_path))
            text_parts = []
            chars_so_far = 0
            for page in doc:
                page_text = page.get_text()
                text_parts.append(page_text)
                chars_so_far += len(page_text)
                if chars_so_far >= max_chars:
                    break
            doc.close()
            text = " ".join(text_parts)[:max_chars].strip()
            if len(text) >= min_chars:
                results[pmid] = text
            else:
                failed += 1
        except Exception:
            failed += 1

    logger.info(f"Extracted text from {len(results)} PDFs ({failed} failed/skipped)")
    return results


def _fetch_pubmed_labels(
    pmids: List[str],
    cache_path: Optional[Path] = None,
    batch_size: int = 200,
) -> Dict[str, str]:
    """
    Fetch PubMed publication types via NCBI E-utilities and classify.

    Classification logic:
      - Has "Systematic Review" OR "Meta-Analysis" → scientific_paper
      - Has "Review" but NOT "Systematic Review"/"Meta-Analysis" → literature_review
      - Everything else → scientific_paper

    Caches results to cache_path (JSON) so we only hit the API once.
    """
    # Load cache if available
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached PubMed labels from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        # Check coverage — only fetch missing PMIDs
        missing = [p for p in pmids if p not in cached]
        if not missing:
            logger.info(f"All {len(pmids)} PMIDs found in cache")
            return cached
        logger.info(f"Cache has {len(cached)} entries, {len(missing)} PMIDs missing")
    else:
        cached = {}
        missing = list(pmids)

    api_key = os.environ.get("NCBI_API_KEY", "")
    rate_limit = 10 if api_key else 3  # requests per second
    delay = 1.0 / rate_limit

    logger.info(f"Fetching PubMed metadata for {len(missing)} PMIDs "
                f"(rate: {rate_limit}/sec, batches of {batch_size})...")

    labels: Dict[str, str] = dict(cached)

    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(params)

        try:
            req = Request(url, headers={"User-Agent": "PubGuard/1.0"})
            with urlopen(req, timeout=30) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                # Extract PMID
                pmid_el = article.find(".//PMID")
                if pmid_el is None or pmid_el.text is None:
                    continue
                pmid = pmid_el.text.strip()

                # Extract PublicationType tags
                pub_types = []
                for pt in article.findall(".//PublicationType"):
                    if pt.text:
                        pub_types.append(pt.text.strip())

                # Classification
                pub_types_lower = [pt.lower() for pt in pub_types]
                has_systematic = any("systematic review" in pt for pt in pub_types_lower)
                has_meta = any("meta-analysis" in pt for pt in pub_types_lower)
                has_review = any(pt == "review" for pt in pub_types_lower)

                if has_systematic or has_meta:
                    labels[pmid] = "scientific_paper"
                elif has_review:
                    labels[pmid] = "literature_review"
                else:
                    labels[pmid] = "scientific_paper"

            logger.info(f"  Fetched batch {i // batch_size + 1} "
                        f"({min(i + batch_size, len(missing))}/{len(missing)})")
        except (URLError, ET.ParseError) as e:
            logger.warning(f"  Batch {i // batch_size + 1} failed: {e}")
            # Label unfetched PMIDs as scientific_paper (safe default)
            for pmid in batch:
                if pmid not in labels:
                    labels[pmid] = "scientific_paper"

        time.sleep(delay)

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(labels, f)
        logger.info(f"Cached {len(labels)} PubMed labels to {cache_path}")

    return labels


def _reconstruct_abstract(inverted_index: Dict) -> str:
    """Reconstruct abstract text from OpenAlex abstract_inverted_index format."""
    positions: Dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[v] for v in sorted(positions.keys()))


def _fetch_openalex_reviews(
    n_reviews: int = 15000,
    cache_path: Optional[Path] = None,
    per_page: int = 200,
) -> List[Dict[str, str]]:
    """
    Fetch review article abstracts from OpenAlex API.

    Uses filter=type:review,has_abstract:true with cursor-based pagination.
    Abstracts are reconstructed from the inverted index format.

    Args:
        n_reviews: Number of review abstracts to fetch
        cache_path: Cache file path (NDJSON) to avoid re-fetching
        per_page: Results per API page (max 200)

    Returns:
        List of {"text": abstract_text, "label": "literature_review"}
    """
    # Check cache — load existing samples and resume cursor if available
    samples: List[Dict[str, str]] = []
    cursor_file = Path(str(cache_path) + ".cursor") if cache_path else None

    if cache_path and cache_path.exists():
        logger.info(f"Loading cached OpenAlex reviews from {cache_path}")
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if len(samples) >= n_reviews:
            logger.info(f"  Loaded {len(samples)} cached reviews (have enough)")
            return samples[:n_reviews]
        logger.info(f"  Cache has {len(samples)}, need {n_reviews} — resuming fetch")

    # Resume from saved cursor, or start fresh
    cursor = "*"
    if cursor_file and cursor_file.exists():
        saved_cursor = cursor_file.read_text().strip()
        if saved_cursor and samples:
            cursor = saved_cursor
            logger.info(f"  Resuming from saved cursor (page position)")

    logger.info(f"Fetching {n_reviews - len(samples)} more review abstracts from OpenAlex API...")

    base_url = (
        "https://api.openalex.org/works?"
        "filter=type:review,has_abstract:true"
        "&select=id,abstract_inverted_index"
        f"&per_page={per_page}"
        "&mailto=pubguard@example.com"
    )
    pages_fetched = 0
    max_pages = (n_reviews // per_page) + 10  # safety margin

    while len(samples) < n_reviews and pages_fetched < max_pages:
        url = f"{base_url}&cursor={cursor}"
        try:
            req = Request(url, headers={"User-Agent": "PubGuard/1.0"})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            for work in data.get("results", []):
                aii = work.get("abstract_inverted_index")
                if not aii:
                    continue
                abstract = _reconstruct_abstract(aii)
                if len(abstract) >= 100:
                    samples.append({"text": abstract, "label": "literature_review"})
                    if len(samples) >= n_reviews:
                        break

            cursor = data.get("meta", {}).get("next_cursor")
            if cursor is None:
                break

            pages_fetched += 1
            if pages_fetched % 10 == 0:
                logger.info(f"  Fetched {len(samples)}/{n_reviews} reviews ({pages_fetched} pages)...")

            # Save cursor after each page so we can resume on failure
            if cursor_file:
                cursor_file.write_text(cursor)

            # Polite rate: ~10 req/sec is fine for OpenAlex
            time.sleep(0.1)

        except (URLError, json.JSONDecodeError) as e:
            logger.warning(f"  OpenAlex fetch error on page {pages_fetched}: {e}")
            # Save what we have so far before potentially failing
            if cache_path and samples:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
                logger.info(f"  Saved {len(samples)} samples to cache after error")
            time.sleep(2)
            pages_fetched += 1
            continue

    logger.info(f"  Total: {len(samples)} review abstracts from OpenAlex")

    # Cache results + clean up cursor file on success
    if cache_path and samples:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        logger.info(f"  Cached to {cache_path}")
        if cursor_file and len(samples) >= n_reviews:
            cursor_file.unlink(missing_ok=True)

    return samples[:n_reviews]


def _download_and_extract_pdf(pdf_url: str, max_chars: int = 4000, timeout: int = 15) -> Optional[str]:
    """Download a PDF from a URL and extract text using PyMuPDF.

    Returns extracted text (up to max_chars) or None on failure.
    """
    import fitz  # PyMuPDF

    try:
        req = Request(pdf_url, headers={"User-Agent": "PubGuard/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            # Read up to 5 MB to avoid memory issues with huge PDFs
            pdf_bytes = resp.read(5 * 1024 * 1024)

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        chars_so_far = 0
        for page in doc:
            page_text = page.get_text()
            text_parts.append(page_text)
            chars_so_far += len(page_text)
            if chars_so_far >= max_chars:
                break
        doc.close()
        text = " ".join(text_parts)[:max_chars].strip()
        return text if len(text) >= 200 else None
    except Exception:
        return None


def _fetch_openalex_review_fulltexts(
    n_reviews: int = 15000,
    cache_path: Optional[Path] = None,
    per_page: int = 200,
    max_workers: int = 10,
) -> List[Dict[str, str]]:
    """
    Fetch full-text review articles from OpenAlex open-access PDFs.

    Queries OpenAlex for OA review articles, downloads PDFs in parallel,
    and extracts text with PyMuPDF. Falls back to abstract when PDF
    download fails. Uses a separate cache from _fetch_openalex_reviews().

    Args:
        n_reviews: Target number of review full-texts to collect
        cache_path: Cache file path (NDJSON) for incremental resumption
        per_page: Results per API page (max 200)
        max_workers: Number of parallel PDF download threads

    Returns:
        List of {"text": full_text_or_abstract, "label": "literature_review"}
    """
    samples: List[Dict[str, str]] = []
    seen_ids: set = set()
    cursor_file = Path(str(cache_path) + ".cursor") if cache_path else None

    # ── Load cache ────────────────────────────────────────────────
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached OpenAlex review full-texts from {cache_path}")
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    samples.append(entry)
                    oa_id = entry.get("_id", "")
                    if oa_id:
                        seen_ids.add(oa_id)
        if len(samples) >= n_reviews:
            logger.info(f"  Loaded {len(samples)} cached full-texts (have enough)")
            return [{"text": s["text"], "label": s["label"]} for s in samples[:n_reviews]]
        logger.info(f"  Cache has {len(samples)}, need {n_reviews} — resuming fetch")

    # ── Resume from saved cursor, or start fresh ──────────────────
    cursor = "*"
    if cursor_file and cursor_file.exists():
        saved_cursor = cursor_file.read_text().strip()
        if saved_cursor and samples:
            cursor = saved_cursor
            logger.info(f"  Resuming from saved cursor (page position)")

    # Oversample 3x since many PDFs will fail (paywalls, redirects, etc.)
    metadata_target = max((n_reviews - len(samples)) * 3, 1000)
    logger.info(f"Fetching OA review metadata from OpenAlex "
                f"(target {n_reviews - len(samples)} full-texts, "
                f"fetching ~{metadata_target} candidates)...")

    base_url = (
        "https://api.openalex.org/works?"
        "filter=type:review,open_access.is_oa:true"
        "&select=id,abstract_inverted_index,best_oa_location"
        f"&per_page={per_page}"
        "&mailto=pubguard@example.com"
    )

    # Collect candidate works with PDF URLs
    candidates: List[dict] = []  # {"id": ..., "pdf_url": ..., "abstract": ...}
    pages_fetched = 0
    max_pages = (metadata_target // per_page) + 10

    while len(candidates) < metadata_target and pages_fetched < max_pages:
        url = f"{base_url}&cursor={cursor}"
        try:
            req = Request(url, headers={"User-Agent": "PubGuard/1.0"})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            for work in data.get("results", []):
                work_id = work.get("id", "")
                if work_id in seen_ids:
                    continue

                # Extract PDF URL from best_oa_location
                oa_loc = work.get("best_oa_location") or {}
                pdf_url = oa_loc.get("pdf_url")

                # Also reconstruct abstract as fallback
                aii = work.get("abstract_inverted_index")
                abstract = _reconstruct_abstract(aii) if aii else ""

                if pdf_url:
                    candidates.append({
                        "id": work_id,
                        "pdf_url": pdf_url,
                        "abstract": abstract,
                    })

            cursor = data.get("meta", {}).get("next_cursor")
            if cursor is None:
                break

            pages_fetched += 1
            if pages_fetched % 20 == 0:
                logger.info(f"  Metadata: {len(candidates)} candidates "
                            f"({pages_fetched} pages)...")

            if cursor_file:
                cursor_file.write_text(cursor)

            time.sleep(0.1)

        except (URLError, json.JSONDecodeError) as e:
            logger.warning(f"  OpenAlex fetch error on page {pages_fetched}: {e}")
            time.sleep(2)
            pages_fetched += 1
            continue

    logger.info(f"  Collected {len(candidates)} OA review candidates with PDF URLs")

    # ── Download PDFs in parallel ─────────────────────────────────
    fulltext_count = 0
    abstract_fallback_count = 0
    failed_count = 0
    batch_start = len(samples)

    def _process_candidate(cand):
        text = _download_and_extract_pdf(cand["pdf_url"])
        if text:
            return {"text": text, "label": "literature_review",
                    "_id": cand["id"], "_source": "fulltext"}
        # Fall back to abstract
        if cand["abstract"] and len(cand["abstract"]) >= 100:
            return {"text": cand["abstract"], "label": "literature_review",
                    "_id": cand["id"], "_source": "abstract_fallback"}
        return None

    logger.info(f"  Downloading PDFs with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_candidate, c): c for c in candidates}

        for future in as_completed(futures):
            if len(samples) >= n_reviews:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            result = future.result()
            if result is None:
                failed_count += 1
                continue

            samples.append(result)
            if result["_source"] == "fulltext":
                fulltext_count += 1
            else:
                abstract_fallback_count += 1

            # Incremental save every 500 successful downloads
            if (len(samples) - batch_start) % 500 == 0 and cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
                logger.info(f"  Progress: {len(samples)}/{n_reviews} "
                            f"(fulltext={fulltext_count}, "
                            f"abstract_fallback={abstract_fallback_count}, "
                            f"failed={failed_count})")

    logger.info(f"  Download complete: {fulltext_count} full-text, "
                f"{abstract_fallback_count} abstract fallbacks, "
                f"{failed_count} failed")
    logger.info(f"  Total review samples: {len(samples)}")

    # ── Cache results ─────────────────────────────────────────────
    if cache_path and samples:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        logger.info(f"  Cached to {cache_path}")
        if cursor_file and len(samples) >= n_reviews:
            cursor_file.unlink(missing_ok=True)

    # Strip internal metadata before returning
    return [{"text": s["text"], "label": s["label"]} for s in samples[:n_reviews]]


def _resolve_poster_pdf_path(json_path: str) -> Optional[Path]:
    """Resolve a poster PDF path from the classification JSON to its local location.

    The JSON stores paths like /home/joneill/vaults/.../poster-pdf-meta/downloads/...
    The actual local copies live at /storage/poster-pdf-meta_downloads/...
    """
    p = Path(json_path)
    if p.exists():
        return p

    # Rewrite: vaults/…/poster-pdf-meta/downloads/X → /storage/poster-pdf-meta_downloads/X
    s = str(p)
    marker = "poster-pdf-meta/downloads/"
    idx = s.find(marker)
    if idx != -1:
        local = Path("/storage/poster-pdf-meta_downloads") / s[idx + len(marker):]
        if local.exists():
            return local

    return None


def _extract_poster_texts(
    poster_corpus: Optional[Path] = None,
    max_posters: int = 15000,
    max_chars: int = 4000,
    min_chars: int = 100,
) -> List[Dict[str, str]]:
    """
    Load real poster texts for training.

    Sources (tried in order):
      1. poster_texts_for_pubguard.ndjson — pre-extracted full-length texts
      2. Verified poster PDFs at /storage/poster-pdf-meta_downloads/
         (28K+ real scientific posters from Zenodo & Figshare)

    Args:
        poster_corpus: Override path for poster PDF downloads directory.
                       Default: /storage/poster-pdf-meta_downloads
        max_posters: Maximum poster samples to return
        max_chars: Truncation length for extracted text
        min_chars: Minimum text length to accept
    """
    import fitz  # PyMuPDF

    samples: List[Dict[str, str]] = []

    # ── Source 1: Pre-extracted NDJSON (instant) ────────────────
    ndjson_paths = [
        Path("/home/joneill/pubverse_brett/poster_sentry/poster_texts_for_pubguard.ndjson"),
        Path.cwd().parent / "poster_sentry" / "poster_texts_for_pubguard.ndjson",
    ]
    for ndjson_path in ndjson_paths:
        if ndjson_path.exists():
            logger.info(f"Loading real poster texts from {ndjson_path}")
            with open(ndjson_path) as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("label") == "poster":
                        text = row["text"][:max_chars]
                        if len(text) >= min_chars:
                            samples.append({"text": text, "label": "poster"})
            logger.info(f"  Loaded {len(samples)} real poster texts from NDJSON")
            break

    if len(samples) >= max_posters:
        return samples[:max_posters]

    # ── Source 2: Extract from 28K verified poster PDFs ─────────
    # Classification results JSON (on Nextcloud metadata path)
    classification_paths = [
        Path("/home/joneill/Nextcloud/vaults/jmind/calmi2/poster_science/poster_classifier/classification_results_20251208_152217.json"),
    ]
    if poster_corpus:
        classification_paths.insert(0, poster_corpus / "classification_results_20251208_152217.json")

    classification_json = None
    for cp in classification_paths:
        if cp.exists():
            classification_json = cp
            break

    # Also check: do the actual poster PDFs exist at /storage/?
    local_poster_dir = poster_corpus or Path("/storage/poster-pdf-meta_downloads")
    if classification_json is None or not local_poster_dir.is_dir():
        if samples:
            logger.info(f"  Using {len(samples)} poster samples from NDJSON (no PDF corpus available)")
        else:
            logger.warning("No poster training data found!")
        return samples

    logger.info(f"Loading poster classification results from {classification_json}")
    with open(classification_json) as f:
        cls_data = json.load(f)

    poster_entries = cls_data.get("posters", [])
    logger.info(f"  {len(poster_entries)} verified poster PDFs in catalog")

    needed = max_posters - len(samples)
    random.shuffle(poster_entries)
    extracted = 0
    failed = 0

    for entry in poster_entries:
        if extracted >= needed:
            break

        pdf_path = _resolve_poster_pdf_path(entry["pdf_path"])
        if pdf_path is None:
            failed += 1
            continue

        try:
            doc = fitz.open(str(pdf_path))
            text_parts = []
            chars_so_far = 0
            for page in doc:
                page_text = page.get_text()
                text_parts.append(page_text)
                chars_so_far += len(page_text)
                if chars_so_far >= max_chars:
                    break
            doc.close()
            text = " ".join(text_parts)[:max_chars].strip()
            if len(text) >= min_chars:
                samples.append({"text": text, "label": "poster"})
                extracted += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    logger.info(f"  Extracted {extracted} additional poster texts from PDFs ({failed} failed)")
    logger.info(f"  Total poster samples: {len(samples)}")
    return samples[:max_posters]


def prepare_doc_type_from_pdf_corpus(
    corpus_dir: Path,
    output_dir: Path,
    n_per_class: int = 15000,
    skip_pubmed: bool = False,
    poster_corpus: Optional[Path] = None,
) -> Path:
    """
    Build doc_type training data from real PDFs + HuggingFace sources.

    Real PDF corpus provides scientific_paper + literature_review samples.
    Real poster PDFs (Zenodo/Figshare) provide poster samples.
    HuggingFace sources provide abstract_only and junk samples.

    Args:
        corpus_dir: Path to PDF directory (e.g. /storage/microbiome_metagenomics_research_pdfs)
        output_dir: Where to write NDJSON output
        n_per_class: Max samples per class
        skip_pubmed: If True, reuse cached PubMed labels (skip API calls)
        poster_corpus: Path to poster PDF corpus (poster-pdf-meta directory)
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "doc_type_train.ndjson"
    cache_path = output_dir / "pubmed_labels.json"
    all_samples: List[Dict[str, str]] = []

    logger.info("=== Preparing doc_type dataset (PDF corpus mode) ===")

    # ── Step 1: Extract text from PDFs ──────────────────────────
    logger.info(f"Extracting text from PDFs in {corpus_dir}...")
    pdf_texts = _extract_pdf_texts(corpus_dir)

    if not pdf_texts:
        logger.error(f"No PDFs extracted from {corpus_dir}. Falling back to HuggingFace-only mode.")
        return prepare_doc_type_dataset(output_dir, n_per_class)

    # ── Step 2: Fetch PubMed labels ─────────────────────────────
    pmids = list(pdf_texts.keys())
    if skip_pubmed and cache_path.exists():
        logger.info("Reusing cached PubMed labels (--skip-pubmed)")
        with open(cache_path) as f:
            pubmed_labels = json.load(f)
    else:
        pubmed_labels = _fetch_pubmed_labels(pmids, cache_path=cache_path)

    # ── Step 3: Build labelled samples from PDFs ────────────────
    pdf_scientific = []
    pdf_review = []
    unlabelled = 0

    for pmid, text in pdf_texts.items():
        label = pubmed_labels.get(pmid)
        if label is None:
            # PMID not in PubMed results — default to scientific_paper
            label = "scientific_paper"
            unlabelled += 1

        sample = {"text": text, "label": label}
        if label == "scientific_paper":
            pdf_scientific.append(sample)
        elif label == "literature_review":
            pdf_review.append(sample)

    logger.info(f"PDF corpus: {len(pdf_scientific)} scientific_paper, "
                f"{len(pdf_review)} literature_review, "
                f"{unlabelled} defaulted (no PubMed match)")

    # ── Step 4: Sample PDF classes ──────────────────────────────
    if len(pdf_scientific) > n_per_class:
        pdf_scientific = random.sample(pdf_scientific, n_per_class)
    all_samples.extend(pdf_scientific)
    logger.info(f"  scientific_paper (PDF): {len(pdf_scientific)}")

    if len(pdf_review) > n_per_class:
        pdf_review = random.sample(pdf_review, n_per_class)
    all_samples.extend(pdf_review)
    logger.info(f"  literature_review (PDF): {len(pdf_review)}")

    # ── Step 4b: Supplement literature_review with OpenAlex ─────
    review_needed = n_per_class - len(pdf_review)
    if review_needed > 0:
        # First: fetch full-text OA review PDFs (preferred — matches inference)
        logger.info(f"Supplementing literature_review with OpenAlex OA full-texts "
                     f"(need {review_needed})...")
        fulltext_cache = output_dir / "openalex_review_fulltexts.ndjson"
        fulltext_reviews = _fetch_openalex_review_fulltexts(
            n_reviews=review_needed,
            cache_path=fulltext_cache,
        )
        all_samples.extend(fulltext_reviews)
        logger.info(f"  literature_review (OpenAlex full-text): {len(fulltext_reviews)}")

        # Second: if still short, supplement with abstract-only reviews
        still_needed = review_needed - len(fulltext_reviews)
        if still_needed > 0:
            logger.info(f"  Still need {still_needed} more — supplementing with "
                         f"OpenAlex abstracts...")
            openalex_cache = output_dir / "openalex_reviews.ndjson"
            abstract_reviews = _fetch_openalex_reviews(
                n_reviews=still_needed,
                cache_path=openalex_cache,
            )
            all_samples.extend(abstract_reviews)
            logger.info(f"  literature_review (OpenAlex abstract fallback): "
                         f"{len(abstract_reviews)}")

    # ── Step 5: Supplement scientific_paper with HuggingFace ────
    hf_needed = n_per_class - len(pdf_scientific)
    if hf_needed > 0:
        logger.info(f"Supplementing scientific_paper with {hf_needed} HuggingFace samples...")
        try:
            ds = load_dataset(
                "armanc/scientific_papers", "arxiv",
                split="train", streaming=True, trust_remote_code=True,
            )
            count = 0
            for row in ds:
                if count >= hf_needed:
                    break
                abstract = row.get("abstract", "") or ""
                article = row.get("article", "") or ""
                text = (abstract + " " + article)[:4000]
                if len(text.strip()) > 100:
                    all_samples.append({"text": text.strip(), "label": "scientific_paper"})
                    count += 1
            logger.info(f"  scientific_paper (HuggingFace supplement): {count}")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace supplement: {e}")

    # ── Step 6: HuggingFace-only classes (poster, abstract_only, junk)
    # abstract_only
    logger.info("Loading gfissore/arxiv-abstracts-2021 for abstract_only...")
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

    # junk
    logger.info("Loading ag_news for junk class...")
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

    # poster — REAL poster data only (no synthetic templates)
    logger.info("Loading real poster data...")
    poster_dir = poster_corpus or Path("/home/joneill/Nextcloud/vaults/jmind/calmi2/poster_science/poster-pdf-meta")
    poster_samples = _extract_poster_texts(poster_dir, max_posters=n_per_class)
    all_samples.extend(poster_samples)
    logger.info(f"  poster (real): {len(poster_samples)}")

    # ── Shuffle and save ─────────────────────────────────────────
    random.shuffle(all_samples)

    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Report distribution
    dist: Dict[str, int] = {}
    for s in all_samples:
        dist[s["label"]] = dist.get(s["label"], 0) + 1
    logger.info(f"Saved {len(all_samples)} samples to {output_path}")
    for label, count in sorted(dist.items()):
        logger.info(f"  {label}: {count}")

    return output_path


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

def prepare_all(
    output_dir: Path,
    n_per_class: int = 15000,
    pdf_corpus: Optional[Path] = None,
    skip_pubmed: bool = False,
    poster_corpus: Optional[Path] = None,
):
    """Download and prepare all three datasets.

    Args:
        output_dir: Where to write training NDJSON files
        n_per_class: Max samples per class
        pdf_corpus: If provided, use real PDF corpus for doc_type data
        skip_pubmed: If True, reuse cached PubMed labels
        poster_corpus: If provided, extract real poster texts from this directory
    """
    output_dir = Path(output_dir)
    logger.info(f"Preparing all datasets in {output_dir}")

    paths = {}
    if pdf_corpus:
        paths["doc_type"] = prepare_doc_type_from_pdf_corpus(
            corpus_dir=pdf_corpus,
            output_dir=output_dir,
            n_per_class=n_per_class,
            skip_pubmed=skip_pubmed,
            poster_corpus=poster_corpus,
        )
    else:
        paths["doc_type"] = prepare_doc_type_dataset(output_dir, n_per_class)
    paths["ai_detect"] = prepare_ai_detect_dataset(output_dir, n_per_class)
    paths["toxicity"] = prepare_toxicity_dataset(output_dir, n_per_class)

    logger.info("All datasets prepared!")
    return paths

#!/usr/bin/env python3
"""
Full training pipeline: download data → train heads → evaluate.

Usage:
    cd /home/joneill/pubverse_brett/pub_check
    source ~/myenv/bin/activate
    pip install -e ".[train]"
    python scripts/train_pubguard.py [--data-dir ./pubguard_data] [--n-per-class 15000]

    # Train on real PDF corpus (adds literature_review class):
    python scripts/train_pubguard.py --pdf-corpus /storage/microbiome_metagenomics_research_pdfs

    # Reuse cached PubMed labels on subsequent runs:
    python scripts/train_pubguard.py --pdf-corpus /storage/microbiome_metagenomics_research_pdfs --skip-pubmed
"""

import argparse
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from pathlib import Path
from pubguard.config import PubGuardConfig
from pubguard.data import prepare_all
from pubguard.train import train_all


def main():
    parser = argparse.ArgumentParser(description="Train PubGuard")
    parser.add_argument("--data-dir", default="./pubguard_data",
                        help="Directory for training data")
    parser.add_argument("--models-dir", default=None,
                        help="Override models output directory")
    parser.add_argument("--n-per-class", type=int, default=15000,
                        help="Samples per class per head")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Held-out test fraction")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (use existing data)")
    parser.add_argument("--pdf-corpus", default=None,
                        help="Path to real PDF corpus directory "
                             "(default: None, uses HuggingFace-only mode)")
    parser.add_argument("--skip-pubmed", action="store_true",
                        help="Reuse cached PubMed labels instead of re-fetching")
    parser.add_argument("--poster-corpus", default=None,
                        help="Path to poster PDF corpus directory "
                             "(default: auto-detect from poster-pdf-meta)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = PubGuardConfig()
    if args.models_dir:
        config.models_dir = Path(args.models_dir)

    pdf_corpus = Path(args.pdf_corpus) if args.pdf_corpus else None
    poster_corpus = Path(args.poster_corpus) if args.poster_corpus else None

    # Step 1: Download and prepare datasets
    if not args.skip_download:
        # Delete ALL stale embeddings caches when retraining from scratch
        cache_dir = data_dir / "embeddings_cache"
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.npy"):
                logging.info(f"Removing stale embeddings cache: {cache_file}")
                cache_file.unlink()

        prepare_all(
            data_dir,
            n_per_class=args.n_per_class,
            pdf_corpus=pdf_corpus,
            skip_pubmed=args.skip_pubmed,
            poster_corpus=poster_corpus,
        )

    # Step 2: Train all heads
    train_all(data_dir, config=config, test_size=args.test_size)

    # Step 3: Quick smoke test
    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    from pubguard import PubGuard

    guard = PubGuard(config=config)
    guard.initialize()

    # Smoke test texts mimic real PDF-extracted text (messy headers, journal
    # metadata, author blocks) — NOT clean markdown-style section labels.
    test_cases = [
        (
            "RESEARCH ARTICLE\nOPEN ACCESS\n"
            "Deep insights into the gut microbial community of a large-scale "
            "metagenomic cohort reveal novel species-level associations with "
            "cardiometabolic health\n"
            "Sarah M. Chen 1,2*, David R. Patel 1, Jun Liu 3, Maria Rodriguez 4, "
            "Emily K. Watson 1\n"
            "1 Department of Microbiology, Stanford University School of Medicine, "
            "Stanford, CA 94305, USA. 2 Chan Zuckerberg Biohub, San Francisco, CA "
            "94158, USA. 3 BGI Genomics, Shenzhen, Guangdong 518083, China. "
            "4 Instituto de Microbiologia, Universidad Nacional Autonoma de Mexico, "
            "Mexico City, Mexico.\n"
            "* Correspondence: smchen@stanford.edu\n\n"
            "Received: 15 March 2024; Accepted: 22 July 2024; Published: 10 August 2024\n\n"
            "The human gut microbiome is a complex ecosystem implicated in metabolic, "
            "immunological, and neurological processes. While prior studies have linked "
            "gut microbial composition to cardiometabolic disease, most relied on 16S "
            "rRNA amplicon sequencing with limited taxonomic resolution. Here we present "
            "a species- and strain-level analysis of 2,847 shotgun metagenomes from "
            "the Global Microbiome Health Initiative (GMHI) cohort, spanning five "
            "continents. Fecal samples were collected following standardized protocols "
            "and sequenced on Illumina NovaSeq 6000 (2x150 bp, median depth 8.2 Gbp). "
            "Taxonomic profiling with MetaPhlAn 4 identified 1,523 species, of which "
            "312 were present in >90% of individuals. We found that Prevotella copri "
            "clade diversity was significantly associated with improved insulin "
            "sensitivity (beta=-0.23, FDR=0.003) independent of BMI and diet. "
            "Conversely, elevated Ruminococcus gnavus abundance correlated with "
            "systemic inflammation markers (hs-CRP, IL-6) and was enriched 2.3-fold "
            "in participants with metabolic syndrome (p<0.001). Functional profiling "
            "with HUMAnN 3 revealed 87 differentially abundant pathways, including "
            "enrichment of butyrate biosynthesis via acetyl-CoA in metabolically "
            "healthy individuals and overrepresentation of lipopolysaccharide "
            "biosynthesis in the metabolic syndrome group. Strain-level analysis "
            "using StrainPhlAn 4 identified geographic subspeciation patterns in "
            "Faecalibacterium prausnitzii consistent with local dietary adaptation. "
            "Our findings demonstrate that species-level resolution substantially "
            "enhances the predictive power of microbiome-based cardiometabolic risk "
            "models (AUC 0.82 vs 0.71 for genus-level, DeLong p=0.004). These "
            "results underscore the clinical potential of high-resolution metagenomics "
            "and suggest specific microbial targets for intervention.",
            "scientific_paper",
        ),
        (
            "JOURNAL OF CLINICAL MICROBIOLOGY REVIEWS\n"
            "0893-8512/24/$04.00+0\nVol. 37, No. 4\n"
            "Copyright 2024, American Society for Microbiology. All Rights Reserved.\n\n"
            "REVIEW\n"
            "The Gut Microbiome in Inflammatory Bowel Disease:\n"
            "A Narrative Review of Two Decades of Research\n\n"
            "James A. O'Brien,1 Priya Sharma,2 and Marcus K. Tanaka3*\n"
            "1 Division of Gastroenterology, Massachusetts General Hospital, Boston, "
            "Massachusetts; 2 Department of Immunology, University of Oxford, Oxford, "
            "United Kingdom; 3 National Institute of Allergy and Infectious Diseases, "
            "NIH, Bethesda, Maryland\n"
            "* Corresponding author. E-mail: tanaka.mk@nih.gov\n\n"
            "The relationship between the intestinal microbiota and inflammatory bowel "
            "disease (IBD) has been a subject of intense investigation since the "
            "advent of culture-independent sequencing methods. In this narrative "
            "review, we survey over 3,000 published studies spanning the period from "
            "2004 to 2024, covering both Crohn's disease (CD) and ulcerative colitis "
            "(UC). We examine the consistent finding of reduced microbial diversity "
            "in IBD patients, with particular attention to the depletion of "
            "Faecalibacterium prausnitzii, Roseburia spp., and other obligate "
            "anaerobes that produce short-chain fatty acids. We discuss the "
            "controversial role of Escherichia coli, particularly adherent-invasive "
            "strains (AIEC), in disease initiation versus perpetuation. "
            "Methodological considerations are addressed, including the impact of "
            "sample collection protocols, sequencing depth, and bioinformatic "
            "pipelines on reported findings. The review evaluates therapeutic "
            "strategies targeting the microbiome, including fecal microbiota "
            "transplantation (FMT), which has shown response rates of 24-65% in "
            "randomized controlled trials for UC but with limited evidence in CD. "
            "We also discuss emerging interventions including defined consortia, "
            "bacteriophage therapy, and dietary modulation. Key knowledge gaps "
            "are identified, including the need for longitudinal multi-omic "
            "studies and improved understanding of host-microbe-metabolite "
            "interactions. We conclude that while substantial progress has been "
            "made, translation into clinical practice requires larger "
            "interventional trials with standardized endpoints.",
            "literature_review",
        ),
        (
            "🎉 POOL PARTY THIS SATURDAY! 🏊 Come join us at the community center "
            "pool. Bring snacks and sunscreen. RSVP to poolparty@gmail.com by Thursday!",
            "junk",
        ),
        (
            "TITLE: Deep Learning for Medical Imaging\nAUTHORS: J. Smith, A. Lee\n"
            "AFFILIATION: MIT, CSAIL\n\nKey Findings:\n• 95% accuracy on chest X-rays\n"
            "• Novel attention mechanism reduces false positives by 30%\n"
            "• Validated on 10,000 images from 3 clinical sites\n\n"
            "Background: Automated detection of pulmonary nodules in chest radiographs "
            "remains challenging due to overlapping structures.\n\n"
            "Methods: We trained a ResNet-50 backbone with custom attention layers on "
            "50K annotated chest X-rays from the MIMIC-CXR dataset.\n\n"
            "Conclusions: Our approach demonstrates clinical-grade performance.\n\n"
            "Contact: jsmith@mit.edu  |  Poster #P-247",
            "poster",
        ),
        (
            "We introduce a memory-efficient attention mechanism that "
            "reduces the quadratic complexity of standard self-attention "
            "to O(n sqrt(n)) while preserving model quality. By partitioning "
            "the input sequence into fixed-size blocks and computing "
            "attention only within and between neighboring blocks, our "
            "approach processes sequences of up to 64K tokens on a single "
            "GPU with 24GB of memory. Experiments on long-document "
            "summarization tasks show that our block-sparse method matches "
            "full attention performance within 0.3 ROUGE-L points while "
            "reducing peak memory usage by 4x and wall-clock training time "
            "by 2.1x.",
            "abstract_only",
        ),
    ]

    for text, expected_type in test_cases:
        verdict = guard.screen(text)
        status = "✅" if verdict["doc_type"]["label"] == expected_type else "⚠️"
        print(f"  {status} Expected: {expected_type:20s} Got: {verdict['doc_type']['label']:20s} "
              f"(score={verdict['doc_type']['score']:.3f})")
        print(f"     AI: {verdict['ai_generated']['label']} ({verdict['ai_generated']['score']:.3f})  "
              f"Toxic: {verdict['toxicity']['label']} ({verdict['toxicity']['score']:.3f})  "
              f"Pass: {verdict['pass']}")

    print(f"\n✅ Training complete! Heads saved to: {config.models_dir}")


if __name__ == "__main__":
    main()

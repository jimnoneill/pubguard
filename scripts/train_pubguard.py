#!/usr/bin/env python3
"""
Full training pipeline: download data → train heads → evaluate.

Usage:
    cd /home/joneill/pubverse_brett/pub_check
    source ~/myenv/bin/activate
    pip install -e ".[train]"
    python scripts/train_pubguard.py [--data-dir ./pubguard_data] [--n-per-class 15000]
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
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = PubGuardConfig()
    if args.models_dir:
        config.models_dir = Path(args.models_dir)

    # Step 1: Download and prepare datasets
    if not args.skip_download:
        prepare_all(data_dir, n_per_class=args.n_per_class)

    # Step 2: Train all heads
    train_all(data_dir, config=config, test_size=args.test_size)

    # Step 3: Quick smoke test
    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    from pubguard import PubGuard

    guard = PubGuard(config=config)
    guard.initialize()

    test_cases = [
        (
            "Introduction: We present a novel deep learning approach for protein "
            "structure prediction. Methods: We trained a transformer model on 50,000 "
            "protein sequences from the PDB database. Results: Our model achieves "
            "state-of-the-art accuracy with an RMSD of 1.2 Å on the CASP14 benchmark. "
            "Discussion: These results demonstrate the potential of attention mechanisms "
            "for structural biology. References: [1] AlphaFold (2021) [2] ESMFold (2022)",
            "scientific_paper",
        ),
        (
            "🎉 POOL PARTY THIS SATURDAY! 🏊 Come join us at the community center "
            "pool. Bring snacks and sunscreen. RSVP to poolparty@gmail.com by Thursday!",
            "junk",
        ),
        (
            "TITLE: Deep Learning for Medical Imaging\nAUTHORS: J. Smith, A. Lee\n"
            "AFFILIATION: MIT\n\nKey Findings:\n• 95% accuracy on chest X-rays\n"
            "• Novel attention mechanism\n\nContact: jsmith@mit.edu",
            "poster",
        ),
        (
            "We investigate the role of microRNAs in hepatocellular carcinoma "
            "progression. Using RNA-seq data from 200 patient samples collected at "
            "three clinical sites, we identified 15 differentially expressed miRNAs "
            "associated with tumor stage (FDR < 0.01).",
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

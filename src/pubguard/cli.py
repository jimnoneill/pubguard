"""
Command-line interface for PubGuard.

Usage:
    # Download datasets and train
    pubguard train --data-dir ./data

    # Download datasets only
    pubguard prepare --data-dir ./data

    # Screen a text file
    pubguard screen input.txt

    # Screen extracted PDF text from stdin
    cat extracted_text.txt | pubguard screen -

    # Batch screen NDJSON
    pubguard batch input.ndjson output.ndjson
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .classifier import PubGuard
from .config import PubGuardConfig


def cmd_prepare(args):
    """Download and prepare training datasets."""
    from .data import prepare_all

    prepare_all(Path(args.data_dir), n_per_class=args.n_per_class)


def cmd_train(args):
    """Prepare data (if needed) and train all heads."""
    from .data import prepare_all
    from .train import train_all

    data_dir = Path(args.data_dir)

    if args.download:
        prepare_all(data_dir, n_per_class=args.n_per_class)

    config = PubGuardConfig()
    if args.models_dir:
        config.models_dir = Path(args.models_dir)

    train_all(data_dir, config=config, test_size=args.test_size)


def cmd_screen(args):
    """Screen a single document."""
    config = PubGuardConfig()
    if args.models_dir:
        config.models_dir = Path(args.models_dir)

    guard = PubGuard(config=config)
    guard.initialize()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(errors="replace")

    verdict = guard.screen(text)

    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        _print_verdict(verdict)


def cmd_batch(args):
    """Batch-screen an NDJSON file."""
    config = PubGuardConfig()
    if args.models_dir:
        config.models_dir = Path(args.models_dir)

    guard = PubGuard(config=config)
    guard.initialize()

    start = time.time()
    processed = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        batch_texts = []
        batch_records = []

        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "") or record.get("abstract", "") or ""
            batch_texts.append(text)
            batch_records.append(record)

            if len(batch_texts) >= config.batch_size:
                verdicts = guard.screen_batch(batch_texts)
                for rec, verd in zip(batch_records, verdicts):
                    rec["pubguard"] = verd
                    fout.write(json.dumps(rec) + "\n")
                processed += len(batch_texts)
                batch_texts, batch_records = [], []

        # Final batch
        if batch_texts:
            verdicts = guard.screen_batch(batch_texts)
            for rec, verd in zip(batch_records, verdicts):
                rec["pubguard"] = verd
                fout.write(json.dumps(rec) + "\n")
            processed += len(batch_texts)

    elapsed = time.time() - start
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"Screened {processed:,} records in {elapsed:.1f}s ({rate:,.0f} rec/s)")
    print(f"Output: {args.output}")


def _print_verdict(v: dict):
    """Pretty-print a verdict."""
    pass_icon = "✅" if v["pass"] else "❌"
    print(f"\n{pass_icon}  PubGuard Verdict: {'PASS' if v['pass'] else 'FAIL'}")
    print(f"   Document type:  {v['doc_type']['label']:20s} (score: {v['doc_type']['score']:.3f})")
    print(f"   AI detection:   {v['ai_generated']['label']:20s} (score: {v['ai_generated']['score']:.3f})")
    print(f"   Toxicity:       {v['toxicity']['label']:20s} (score: {v['toxicity']['score']:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PubGuard — Scientific Publication Gatekeeper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help="Override models directory",
    )

    subparsers = parser.add_subparsers(dest="command")

    # prepare
    p_prepare = subparsers.add_parser("prepare", help="Download and prepare datasets")
    p_prepare.add_argument("--data-dir", default="./pubguard_data")
    p_prepare.add_argument("--n-per-class", type=int, default=15000)

    # train
    p_train = subparsers.add_parser("train", help="Train classification heads")
    p_train.add_argument("--data-dir", default="./pubguard_data")
    p_train.add_argument("--models-dir", default=None)
    p_train.add_argument("--download", action="store_true", default=True,
                        help="Download datasets before training")
    p_train.add_argument("--no-download", action="store_false", dest="download")
    p_train.add_argument("--n-per-class", type=int, default=15000)
    p_train.add_argument("--test-size", type=float, default=0.15)

    # screen
    p_screen = subparsers.add_parser("screen", help="Screen a single document")
    p_screen.add_argument("input", help="Text file to screen (or - for stdin)")
    p_screen.add_argument("--json", action="store_true", help="JSON output")

    # batch
    p_batch = subparsers.add_parser("batch", help="Batch screen NDJSON")
    p_batch.add_argument("input", help="Input NDJSON file")
    p_batch.add_argument("output", help="Output NDJSON file")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "screen":
        cmd_screen(args)
    elif args.command == "batch":
        cmd_batch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

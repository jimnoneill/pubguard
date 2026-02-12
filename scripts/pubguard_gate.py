#!/usr/bin/env python3
"""
PubGuard gate for run_pubverse_pipeline.sh integration.

Reads extracted PDF text from stdin or a file, screens it, and:
  - Prints the error code string to STDOUT (always, for pipeline capture)
  - Prints verdict JSON and diagnostics to STDERR
  - Exits 0 (pass) → pipeline continues
  - Exits 1 (fail) → pipeline halts with error code

Error code format:
    PV-0[doc_type][ai_detect][toxicity] | NAME | snarky message
    PV-0000 = scientific_paper + human + clean = PASS

Usage in run_pubverse_pipeline.sh:
    PUBGUARD_RESULT=$(echo "$PDF_TEXT" | python3 pub_check/scripts/pubguard_gate.py)
    PUBGUARD_EXIT=$?
    echo "$PUBGUARD_RESULT"   # Error code line on stdout

Environment variables:
    PUBGUARD_MODELS_DIR  – Override models directory
    PUBGUARD_STRICT      – Set to "0" to warn instead of gate (exit 0 always)
"""

import json
import sys
import os
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from pubguard import PubGuard, PubGuardConfig
from pubguard.errors import (
    build_pubguard_error,
    empty_input_error,
    gate_bypassed,
)


def main():
    # Read text from stdin or file argument
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        with open(sys.argv[1], errors="replace") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        err = empty_input_error()
        print(str(err))  # stdout: error code line
        print(json.dumps(err.to_dict()), file=sys.stderr)
        sys.exit(1)

    # Configure
    config = PubGuardConfig()
    strict = os.environ.get("PUBGUARD_STRICT", "1") != "0"

    # Screen
    guard = PubGuard(config=config)
    guard.initialize()
    verdict = guard.screen(text)

    # Build structured error code from verdict
    err = build_pubguard_error(verdict)

    # STDOUT: always print the error code line (pipeline captures this)
    print(str(err))

    # STDERR: full verdict JSON for debugging
    print(json.dumps(verdict), file=sys.stderr)

    # Gate decision
    if verdict["pass"]:
        print(f"PUBGUARD: PASS ({err.code})", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"PUBGUARD: FAIL ({err.code})", file=sys.stderr)

        if strict and err.fatal:
            sys.exit(1)
        elif not strict:
            bypass = gate_bypassed()
            print(str(bypass))  # Also print bypass code to stdout
            print(f"PUBGUARD: {bypass.message}", file=sys.stderr)
            sys.exit(0)
        else:
            # Non-fatal flag (e.g. AI detection, toxicity) — warn but pass
            print(f"PUBGUARD: WARNING (non-fatal flag, proceeding)", file=sys.stderr)
            sys.exit(0)


if __name__ == "__main__":
    main()

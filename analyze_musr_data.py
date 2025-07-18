#!/usr/bin/env python3
"""
MuSR Murder Mystery Dataset Analyzer
-----------------------------------
This script gives you a quick statistical snapshot of the **MuSR** murder‑mystery
dataset (gold stories + golden chains‑of‑thought).

What it reports
===============
* **Total mysteries** — number of top‑level examples in the JSON file.
* **Suspect‑count distribution** — how many stories have 1 suspect, 2 suspects …
  etc.
* **Story length stats** (words) — mean / median / min / max.
* **Chain‑of‑thought length stats** (tokens) — mean / median / min / max.

Usage
-----
```bash
python musr_murder_mystery_analysis.py            # default path
python musr_murder_mystery_analysis.py -f path/to/murder_mystery.json
```

By default the script expects the MuSR repo layout and will look for the file
`datasets/murder_mystery.json` relative to the location of this script.

Prerequisites: only the Python standard library is used, so no extra packages
are required.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import statistics


def load_data(path: Path):
    """Read JSON and return as Python object."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(dataset):
    """Compute and print summary statistics for the dataset."""

    total = len(dataset)
    suspect_counter = Counter()
    story_lengths = []
    cot_lengths = []

    for example in dataset:
        # --- suspects -------------------------------------------------------
        suspects = example.get("suspects") or example.get("suspect_infos") or []
        suspect_counter[len(suspects)] += 1

        # --- story text -----------------------------------------------------
        story_text = (
            example.get("story")
            or example.get("chapter")
            or example.get("context")
            or ""
        )
        story_lengths.append(len(story_text.split()))

        # --- golden chain‑of‑thought ---------------------------------------
        cot_text = (
            example.get("golden_chain")
            or example.get("gold_rationale")
            or example.get("chain_of_thought")
            or ""
        )
        if isinstance(cot_text, list):
            cot_text = " ".join(cot_text)
        cot_lengths.append(len(str(cot_text).split()))

    # --- output ------------------------------------------------------------
    print("========== MuSR Murder‑Mystery Dataset Stats ==========")
    print(f"Total mysteries: {total}\n")

    print("Suspect count distribution:")
    for k in sorted(suspect_counter):
        print(f"  {k} suspects: {suspect_counter[k]}")

    if story_lengths:
        print("\nStory length (words):")
        print(
            f"  mean {statistics.mean(story_lengths):.1f} | median {statistics.median(story_lengths)} | "
            f"min {min(story_lengths)} | max {max(story_lengths)}"
        )

    if cot_lengths:
        print("\nChain‑of‑Thought length (tokens):")
        print(
            f"  mean {statistics.mean(cot_lengths):.1f} | median {statistics.median(cot_lengths)} | "
            f"min {min(cot_lengths)} | max {max(cot_lengths)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze MuSR murder‑mystery dataset")
    default_file = Path(__file__).resolve().parent.parent / "datasets" / "murder_mystery.json"
    parser.add_argument("-f", "--file", type=Path, default=default_file, help="Path to murder_mystery.json")
    args = parser.parse_args()

    if not args.file.exists():
        parser.error(f"File not found: {args.file}")

    data = load_data(args.file)
    summarize(data)


if __name__ == "__main__":
    main()

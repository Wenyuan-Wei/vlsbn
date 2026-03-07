#!/usr/bin/env python3
"""
Fetch all matching PDB IDs from RCSB and write them to a text file.

This is Stage 0 of the HPC pipeline.  Run once before submitting the
array job so every task knows its exact batch.

Usage
-----
    python scripts/hpc/fetch_ids.py
    python scripts/hpc/fetch_ids.py --out data/pdb_ids.txt --max-resolution 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlsbn.constants import MAX_RESOLUTION, MAX_RFREE
from vlsbn.pipeline.fetch import fetch_pdb_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch RCSB PDB IDs")
    p.add_argument("--out", type=Path, default=Path("data/pdb_ids.txt"),
                   help="Output text file (one ID per line)")
    p.add_argument("--max-resolution", type=float, default=MAX_RESOLUTION)
    p.add_argument("--max-rfree",      type=float, default=MAX_RFREE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ids = fetch_pdb_ids(args.max_resolution, args.max_rfree)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(ids) + "\n")
    print(f"Wrote {len(ids)} IDs → {args.out}")
    # Print the array size for the recommended batch size (250 PDBs/task)
    for batch in (100, 250, 500):
        n_tasks = (len(ids) + batch - 1) // batch
        print(f"  batch={batch:4d} → {n_tasks:6d} array tasks")


if __name__ == "__main__":
    main()

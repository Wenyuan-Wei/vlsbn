#!/usr/bin/env python3
"""
Fetch all matching PDB IDs from RCSB and write them to disk.

This is Stage 0 of the HPC pipeline.  Run once (on the login node or as a
short Slurm job) before submitting the processing array so every task knows
its exact PDB ID.

Outputs
-------
--out   (default data/pdb_ids.txt)
    Plain-text file, one PDB ID per line.  Used by the array job:
      PDB_ID=$(sed -n "$((TASK_ID + 1))p" data/pdb_ids.txt)

--out-json  (default data/pdb_ids.json)
    JSON file with fetch metadata + the full ID list.  Useful for
    record-keeping and reproducing the exact dataset later.
    Suppress with --no-json.

Usage
-----
    python scripts/hpc/fetch_ids.py
    python scripts/hpc/fetch_ids.py --out data/pdb_ids.txt --max-resolution 2.0
    python scripts/hpc/fetch_ids.py --no-json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlsbn.constants import MAX_RESOLUTION, MAX_RFREE
from vlsbn.pipeline.fetch import fetch_pdb_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch RCSB PDB IDs")
    p.add_argument("--out", type=Path, default=Path("data/pdb_ids.txt"),
                   help="Output text file (one ID per line)")
    p.add_argument("--out-json", type=Path, default=Path("data/pdb_ids.json"),
                   help="Output JSON file with fetch metadata + IDs")
    p.add_argument("--no-json", action="store_true",
                   help="Skip writing the JSON file")
    p.add_argument("--max-resolution", type=float, default=MAX_RESOLUTION)
    p.add_argument("--max-rfree",      type=float, default=MAX_RFREE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ids = fetch_pdb_ids(args.max_resolution, args.max_rfree)
    n   = len(ids)

    # Write plain-text list (used by the array job via sed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(ids) + "\n")
    logger.info("Wrote %d IDs → %s", n, args.out)

    # Write JSON with fetch metadata
    if not args.no_json:
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "max_resolution_angstrom": args.max_resolution,
                "max_rfree":               args.max_rfree,
            },
            "n_ids": n,
            "pdb_ids": ids,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info("Wrote metadata → %s", args.out_json)

    # Array-task planning summary (1 PDB per task)
    print(f"\nFetched {n} PDB IDs.")
    print("\nTo submit the processing array (one task per PDB):")
    print(f"  sbatch --array=0-{n - 1} slurm/01_process.sh")
    print()
    print("Capped submission (recommended — adjust %N to your cluster limit):")
    print(f"  sbatch --array=0-{n - 1}%200 slurm/01_process.sh")


if __name__ == "__main__":
    main()

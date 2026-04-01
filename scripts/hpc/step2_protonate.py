#!/usr/bin/env python3
"""
Step 3: Add hydrogen atoms to all temp PDBs listed in the manifest.

Reads:   data/work/<pdb_id>/manifest.tsv
Writes:  data/work/<pdb_id>/<ligand_id>_h.pdb  for each complex

If protonation fails for a complex the script logs a warning but continues.
The getcontacts step should fall back to the non-H PDB in that case.

Usage
-----
    python scripts/hpc/step2_protonate.py \
        --pdb-id   1abc \
        --work-dir data/work
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlsbn.pipeline.contacts import _add_hydrogens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: protonate temp PDBs")
    p.add_argument("--pdb-id",   required=True,
                   help="PDB ID (used to locate work subdirectory)")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"),
                   help="Root directory for per-PDB work subdirectories")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdb_id = args.pdb_id.strip().lower()
    work_dir = args.work_dir / pdb_id
    manifest = work_dir / "manifest.tsv"

    if not manifest.exists():
        logger.error("Manifest not found: %s — run step1_prepare.py first", manifest)
        sys.exit(1)

    lines = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
    data_rows = lines[1:]  # skip header

    if not data_rows:
        logger.info("No complexes in manifest — nothing to protonate.")
        return

    n_ok = 0
    n_fail = 0
    for row in data_rows:
        cols = row.split("\t")
        ligand_id = cols[0]
        src  = work_dir / f"{ligand_id}.pdb"
        dest = work_dir / f"{ligand_id}_h.pdb"

        if not src.exists():
            logger.warning("Temp PDB not found: %s — skipping", src.name)
            n_fail += 1
            continue

        ok = _add_hydrogens(src, dest)
        if ok:
            n_h = sum(
                1 for line in dest.read_text().splitlines()
                if line.startswith(("ATOM", "HETATM"))
                and line[76:78].strip().upper() == "H"
            )
            logger.info("  %-30s → %-34s  H atoms added: %d",
                        src.name, dest.name, n_h)
            n_ok += 1
        else:
            logger.warning(
                "  %s — protonation failed; getcontacts will use non-H PDB", ligand_id
            )
            n_fail += 1

    logger.info("Protonation done: %d ok, %d failed", n_ok, n_fail)


if __name__ == "__main__":
    main()

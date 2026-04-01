#!/usr/bin/env python3
"""
Step 1+2: Download PDB, parse complexes, write per-complex temp PDBs, delete original.

One array task calls this script for a single PDB ID.  Outputs:

  data/work/<pdb_id>/manifest.tsv
      Header: ligand_id  ligand_resname  n_prot_atoms  n_lig_atoms
      One row per valid protein–ligand complex found in the structure.

  data/work/<pdb_id>/<ligand_id>.pdb
      Stripped PDB containing only protein ATOM records + ligand HETATM
      records, no hydrogen atoms.  One file per complex.

The original downloaded PDB is deleted at the end to save disk space.

Usage
-----
    python scripts/hpc/step1_prepare.py \
        --pdb-id  1abc \
        --work-dir data/work \
        --raw-dir  data/raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlsbn.pipeline.contacts import _write_temp_pdb
from vlsbn.pipeline.fetch import download_pdb
from vlsbn.pipeline.parse import parse_pdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 1+2: download PDB, parse complexes, write temp PDBs"
    )
    p.add_argument("--pdb-id",   required=True,
                   help="PDB ID to process (e.g. 1abc)")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"),
                   help="Root directory for per-PDB work subdirectories")
    p.add_argument("--raw-dir",  type=Path, default=Path("data/raw"),
                   help="Temporary directory for downloaded PDB/CIF files")
    p.add_argument("--delay",    type=float, default=0.05,
                   help="Sleep after download in seconds (RCSB rate-limiting)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdb_id = args.pdb_id.strip().lower()

    out_dir = args.work_dir / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "manifest.tsv"

    # ------------------------------------------------------------------ #
    # Step 1: Download                                                     #
    # ------------------------------------------------------------------ #
    logger.info("Downloading %s", pdb_id)
    pdb_path = download_pdb(pdb_id, dest_dir=args.raw_dir, delay=args.delay)
    logger.info("  Saved → %s", pdb_path)

    # ------------------------------------------------------------------ #
    # Step 1 (cont): Parse                                                 #
    # ------------------------------------------------------------------ #
    logger.info("Parsing %s", pdb_path.name)
    complexes = parse_pdb(pdb_path)
    logger.info("  Found %d valid complex(es)", len(complexes))

    if not complexes:
        logger.warning("No valid complexes in %s — writing empty manifest", pdb_id)
        manifest.write_text("ligand_id\tligand_resname\tn_prot_atoms\tn_lig_atoms\n")
        pdb_path.unlink(missing_ok=True)
        return

    # ------------------------------------------------------------------ #
    # Step 2: Write per-complex temp PDBs + manifest                      #
    # ------------------------------------------------------------------ #
    with manifest.open("w") as mf:
        mf.write("ligand_id\tligand_resname\tn_prot_atoms\tn_lig_atoms\n")
        for c in complexes:
            temp_pdb = out_dir / f"{c.ligand_id}.pdb"
            _write_temp_pdb(c, temp_pdb)
            mf.write(
                f"{c.ligand_id}\t{c.ligand_resname}\t"
                f"{len(c.protein_atoms)}\t{len(c.ligand_atoms)}\n"
            )
            logger.info(
                "  Wrote %-30s  prot=%d  lig=%d",
                temp_pdb.name, len(c.protein_atoms), len(c.ligand_atoms),
            )

    # ------------------------------------------------------------------ #
    # Step 2 (cont): Delete original PDB to save disk                     #
    # ------------------------------------------------------------------ #
    pdb_path.unlink(missing_ok=True)
    logger.info("Deleted %s (disk reclaimed)", pdb_path.name)
    logger.info("Manifest written → %s (%d complex(es))", manifest, len(complexes))


if __name__ == "__main__":
    main()

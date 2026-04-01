#!/usr/bin/env python3
"""
Step 5: Parse getcontacts TSV output → raw contacts CSV.

Reads:   data/work/<pdb_id>/<ligand_id>_contacts.tsv   (one per complex)
Writes:  data/raw_contacts/<pdb_id>_<ligand_id>.csv
         Columns: pdb_id, ligand_id, itype, prot_atom, lig_atom

One output CSV per complex.  The merge step later concatenates all CSVs.
Missing TSV files (e.g. getcontacts failed for a complex) are skipped with
a warning so the script always exits 0 if the manifest is readable.

Usage
-----
    python scripts/hpc/step3_parse_contacts.py \
        --pdb-id   1abc \
        --work-dir data/work \
        --out-dir  data/raw_contacts
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# getcontacts hb subtypes — all normalised to "hb" in output
_HB_ITYPES = {"hbbb", "hbss", "hbsb"}

_CSV_FIELDS = ["pdb_id", "ligand_id", "itype", "prot_atom", "lig_atom"]


def _parse_tsv(tsv_path: Path, pdb_id: str, ligand_id: str) -> list[dict]:
    """Parse one getcontacts TSV → list of raw contact dicts."""
    rows: list[dict] = []
    for line in tsv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        itype = parts[1]
        if itype in _HB_ITYPES or (itype == "hb" and len(parts) >= 5):
            # hb* format: frame  itype  donor_heavy  hydrogen  acceptor_heavy
            atom1, atom2 = parts[2], parts[4]
            itype = "hb"           # normalise subtypes → single column
        else:
            atom1, atom2 = parts[2], parts[3]
        rows.append({
            "pdb_id":    pdb_id,
            "ligand_id": ligand_id,
            "itype":     itype,
            "prot_atom": atom1,
            "lig_atom":  atom2,
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 5: parse getcontacts TSV → raw contacts CSV"
    )
    p.add_argument("--pdb-id",   required=True,
                   help="PDB ID (used to locate work subdirectory and name outputs)")
    p.add_argument("--work-dir", type=Path, default=Path("data/work"),
                   help="Root directory for per-PDB work subdirectories")
    p.add_argument("--out-dir",  type=Path, default=Path("data/raw_contacts"),
                   help="Output directory for raw contacts CSV files")
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
        logger.info("No complexes in manifest — nothing to parse.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_contacts = 0
    for row in data_rows:
        cols = row.split("\t")
        ligand_id = cols[0]
        tsv_path  = work_dir / f"{ligand_id}_contacts.tsv"

        if not tsv_path.exists():
            logger.warning("Contacts TSV not found: %s — skipping", tsv_path.name)
            continue

        contacts = _parse_tsv(tsv_path, pdb_id, ligand_id)
        out_csv  = args.out_dir / f"{pdb_id}_{ligand_id}.csv"

        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(contacts)

        logger.info("  %-40s → %s  (%d contacts)",
                    tsv_path.name, out_csv.name, len(contacts))
        total_contacts += len(contacts)

    logger.info("Done. Total contacts written: %d", total_contacts)


if __name__ == "__main__":
    main()

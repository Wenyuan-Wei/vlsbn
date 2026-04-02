#!/usr/bin/env python3
"""
Step 6: work PDB + raw contacts CSV → contact-density feature parquet.

For each complex listed in manifest.tsv this script:
  1. Re-parses data/work/<pdb_id>/<ligand_id>.pdb to recover atom types
     and coordinates (needed for the possible-pairs denominator).
  2. Reads the raw contacts CSV produced by step3_parse_contacts.
  3. Computes the same normalised contact-density feature vector that
     compute_features() would produce, but without re-running getcontacts.
  4. Writes a single-PDB parquet to data/features/<pdb_id>.parquet
     (one row per complex, meta columns pdb_id + ligand_id + feature cols).

The merge step (merge_train.py) then concatenates all feature parquets.

Usage
-----
    python scripts/hpc/step4_featurize.py \
        --pdb-id   1abc \
        --work-dir data/work \
        --contacts-dir data/raw_contacts \
        --out-dir  data/features
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlsbn.constants import INTERACTION_TYPES, SHELL_RADIUS
from vlsbn.pipeline.atomtypes import LIGAND_ATOM_TYPES, PROTEIN_ATOM_TYPES
from vlsbn.pipeline.contacts import _atom_name_to_index, _count_possible_pairs
from vlsbn.pipeline.parse import parse_pdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ZERO = 0.0  # same sentinel as contacts.py


def _featurize(
    complex_,
    contacts_csv: Path,
    shell_radius: float = SHELL_RADIUS,
    itypes: tuple[str, ...] = INTERACTION_TYPES,
) -> dict[str, float]:
    """Compute contact-density features from pre-parsed contacts CSV."""
    prot_idx = _atom_name_to_index(complex_.protein_atoms)
    lig_idx  = _atom_name_to_index(complex_.ligand_atoms)
    possible = _count_possible_pairs(
        complex_.protein_atoms, complex_.ligand_atoms, shell_radius
    )

    observed: dict[tuple[str, str, str], int] = defaultdict(int)

    if contacts_csv.exists():
        with contacts_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                itype    = row["itype"]
                prot_key = row["prot_atom"]
                lig_key  = row["lig_atom"]

                pi = prot_idx.get(prot_key)
                li = lig_idx.get(lig_key)
                if pi is None or li is None:
                    # getcontacts occasionally swaps atom order
                    pi = prot_idx.get(lig_key)
                    li = lig_idx.get(prot_key)
                if pi is None or li is None:
                    continue

                ptype = complex_.protein_atoms[pi].atom_type
                ltype = complex_.ligand_atoms[li].atom_type
                observed[(ptype, ltype, itype)] += 1
    else:
        logger.warning("Contacts CSV not found: %s — features will be zero", contacts_csv.name)

    features: dict[str, float] = {}
    for ptype, ltype, itype in product(PROTEIN_ATOM_TYPES, LIGAND_ATOM_TYPES, itypes):
        key   = f"{ptype}__{ltype}__{itype}"
        denom = possible.get((ptype, ltype), 0)
        features[key] = (
            _ZERO if denom == 0
            else observed.get((ptype, ltype, itype), 0) / denom
        )

    return features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 6: raw contacts + work PDB → contact-density feature parquet"
    )
    p.add_argument("--pdb-id",       required=True,
                   help="PDB ID (e.g. 1abc)")
    p.add_argument("--work-dir",     type=Path, default=Path("data/work"),
                   help="Root directory for per-PDB work subdirectories")
    p.add_argument("--contacts-dir", type=Path, default=Path("data/raw_contacts"),
                   help="Directory containing raw contacts CSVs from step3_parse_contacts")
    p.add_argument("--out-dir",      type=Path, default=Path("data/features"),
                   help="Output directory for per-PDB feature parquets")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdb_id   = args.pdb_id.strip().lower()
    work_dir = args.work_dir / pdb_id
    manifest = work_dir / "manifest.tsv"

    if not manifest.exists():
        logger.error("Manifest not found: %s — run step1_prepare.py first", manifest)
        sys.exit(1)

    lines     = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
    data_rows = lines[1:]  # skip header

    if not data_rows:
        logger.info("Empty manifest for %s — nothing to featurize.", pdb_id)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for row in data_rows:
        cols      = row.split("\t")
        ligand_id = cols[0]
        work_pdb  = work_dir / f"{ligand_id}.pdb"

        if not work_pdb.exists():
            logger.warning("Work PDB not found: %s — skipping", work_pdb.name)
            continue

        # Re-parse the per-complex PDB to recover atom types + coordinates.
        # parse_pdb() works on the stripped PDB written by _write_temp_pdb.
        complexes = parse_pdb(work_pdb)
        if not complexes:
            logger.warning("parse_pdb returned no complexes for %s — skipping", work_pdb.name)
            continue

        # The work PDB contains exactly one complex.
        c = complexes[0]

        contacts_csv = args.contacts_dir / f"{pdb_id}_{ligand_id}.csv"
        feats = _featurize(c, contacts_csv)
        feats["pdb_id"]    = pdb_id
        feats["ligand_id"] = ligand_id
        rows.append(feats)

        logger.info(
            "  %-30s  prot=%d  lig=%d  contacts_csv=%s",
            ligand_id,
            len(c.protein_atoms),
            len(c.ligand_atoms),
            "found" if contacts_csv.exists() else "MISSING",
        )

    if rows:
        df = pd.DataFrame(rows).fillna(_ZERO)
        meta = [c for c in ("pdb_id", "ligand_id") if c in df.columns]
        feat = [c for c in df.columns if c not in set(meta)]
        df   = df[meta + feat]
        out  = args.out_dir / f"{pdb_id}.parquet"
        df.to_parquet(out, index=False)
        logger.info("Wrote %d row(s) → %s", len(rows), out)
    else:
        logger.warning("No features extracted for %s.", pdb_id)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug script: trace HB contacts through the pipeline for a single PDB.

Checks:
  1. Raw getcontacts output — does it contain hb* lines?
  2. Parsed contacts list — do hb entries survive parsing?
  3. Feature dict — is the hb density > 0 for any atom-type pair?

Usage:
    python scripts/debug_hb.py data/raw/1a4e.pdb
    python scripts/debug_hb.py data/raw/1a4e.pdb --list-itypes
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlsbn.pipeline.contacts import (
    GETCONTACTS_PYTHON,
    GETCONTACTS_SCRIPT,
    _OBABEL,
    _REDUCE,
    _add_hydrogens,
    _run_getcontacts,
    _write_temp_pdb,
)
from vlsbn.pipeline.parse import parse_pdb
from vlsbn.constants import INTERACTION_TYPES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug HB contact detection")
    p.add_argument("pdb", type=Path, help="PDB file to analyse")
    p.add_argument("--list-itypes", action="store_true",
                   help="Also query getcontacts --help to show supported itypes")
    p.add_argument("--ligand-index", type=int, default=0,
                   help="Which ligand complex to use (default: first)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_itypes:
        print("=== getcontacts supported --itypes ===")
        result = subprocess.run(
            [GETCONTACTS_PYTHON, GETCONTACTS_SCRIPT, "--help"],
            capture_output=True, text=True
        )
        for line in (result.stdout + result.stderr).splitlines():
            if "itype" in line.lower() or "hb" in line.lower():
                print(" ", line)
        print()

    # ------------------------------------------------------------------ #
    # Step 1: Parse PDB                                                   #
    # ------------------------------------------------------------------ #
    print(f"=== Parsing {args.pdb.name} ===")
    complexes = parse_pdb(args.pdb)
    print(f"  Found {len(complexes)} complex(es): "
          f"{[c.ligand_id for c in complexes]}")

    if not complexes:
        print("  ERROR: no complexes parsed — check artifact filter / heavy atom count")
        sys.exit(1)

    c = complexes[args.ligand_index]
    print(f"  Using: {c.ligand_id}  (resname={c.ligand_resname})")
    print(f"  Protein atoms: {len(c.protein_atoms)}  |  Ligand atoms: {len(c.ligand_atoms)}")

    # ------------------------------------------------------------------ #
    # Step 2: Write temp PDB, add H atoms, run getcontacts (hb only)     #
    # ------------------------------------------------------------------ #
    print(f"\n=== H-addition tools ===")
    print(f"  obabel : {_OBABEL or '*** NOT FOUND — install conda-forge::openbabel ***'}")
    print(f"  reduce : {_REDUCE or 'not found'}")

    tmpdir = tempfile.mkdtemp()
    pdb_path = Path(tmpdir) / f"{c.ligand_id}.pdb"
    _write_temp_pdb(c, pdb_path)
    print(f"\n=== Temp PDB written (no H): {pdb_path} ===")

    # Count H atoms in original PDB
    n_h_before = sum(
        1 for line in pdb_path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[76:78].strip().upper() == "H"
    )
    print(f"  H atoms before protonation: {n_h_before}")

    # Add H atoms
    h_pdb_path = Path(tmpdir) / f"{c.ligand_id}_h.pdb"
    h_ok = _add_hydrogens(pdb_path, h_pdb_path)
    if h_ok:
        n_h_after = sum(
            1 for line in h_pdb_path.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM")) and line[76:78].strip().upper() == "H"
        )
        print(f"  H atoms after  protonation: {n_h_after}")
        gc_input = h_pdb_path
    else:
        print("  *** Protonation failed — running getcontacts on non-H PDB ***")
        gc_input = pdb_path

    # Run getcontacts requesting hb explicitly (will expand to hbbb/hbss/hbsb)
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [
        GETCONTACTS_PYTHON, GETCONTACTS_SCRIPT,
        "--structure",  str(gc_input),
        "--output",     str(out_path),
        "--itypes",     "hb",
        "--sele",       "protein",
        "--sele2",      f"resname {c.ligand_resname}",
    ]
    print(f"\n  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(f"  Return code: {result.returncode}")
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr[:2000]}")

    # ------------------------------------------------------------------ #
    # Step 3: Inspect raw TSV output                                      #
    # ------------------------------------------------------------------ #
    print(f"\n=== Raw getcontacts output ({out_path}) ===")
    raw_lines = out_path.read_text().splitlines() if out_path.exists() else []
    print(f"  Total lines: {len(raw_lines)}")

    hb_lines = [l for l in raw_lines if not l.startswith("#") and l.strip()
                and l.split("\t")[1:2] and l.split("\t")[1].startswith("hb")]
    all_contact_lines = [l for l in raw_lines if not l.startswith("#") and l.strip()]

    print(f"  Contact lines (non-comment): {len(all_contact_lines)}")
    print(f"  HB lines (itype starts with 'hb'): {len(hb_lines)}")

    if hb_lines:
        print("  First 5 HB lines:")
        for l in hb_lines[:5]:
            print("   ", l)
    else:
        print("  *** NO HB LINES FOUND IN RAW OUTPUT ***")
        if all_contact_lines:
            print("  Unique itypes present:")
            itypes = set()
            for l in all_contact_lines:
                parts = l.split("\t")
                if len(parts) > 1:
                    itypes.add(parts[1])
            for it in sorted(itypes):
                print(f"    {it}")

    # ------------------------------------------------------------------ #
    # Step 4: Run full pipeline with all INTERACTION_TYPES                #
    # ------------------------------------------------------------------ #
    print(f"\n=== Running _run_getcontacts with INTERACTION_TYPES={INTERACTION_TYPES} ===")
    contacts = _run_getcontacts(gc_input, c.ligand_resname, INTERACTION_TYPES)
    print(f"  Total parsed contacts: {len(contacts)}")

    hb_contacts = [(it, a1, a2) for it, a1, a2 in contacts if it == "hb"]
    print(f"  HB contacts after normalisation: {len(hb_contacts)}")
    if hb_contacts:
        print("  First 5:")
        for it, a1, a2 in hb_contacts[:5]:
            print(f"    {it}  {a1}  →  {a2}")
    else:
        print("  *** NO HB CONTACTS AFTER PARSING ***")

    itype_counts: dict[str, int] = {}
    for it, _, _ in contacts:
        itype_counts[it] = itype_counts.get(it, 0) + 1
    print("  All itype counts:", itype_counts)

    # ------------------------------------------------------------------ #
    # Step 5: Check feature dict                                          #
    # ------------------------------------------------------------------ #
    from vlsbn.pipeline.contacts import compute_features
    print(f"\n=== Feature dict (hb columns only) ===")
    feats = compute_features(c)
    hb_feats = {k: v for k, v in feats.items() if "__hb" in k and v > 0}
    print(f"  Non-zero hb features: {len(hb_feats)}")
    if hb_feats:
        for k, v in sorted(hb_feats.items(), key=lambda x: -x[1])[:10]:
            print(f"    {k}: {v:.4f}")
    else:
        print("  *** ALL HB FEATURES ARE ZERO ***")

    # Cleanup
    out_path.unlink(missing_ok=True)
    pdb_path.unlink(missing_ok=True)
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Contact detection and feature-vector computation.

Workflow per complex
--------------------
1. Write a temporary PDB containing only the protein and ligand of interest.
2. Call ``get_static_contacts.py`` (getcontacts) via subprocess to detect
   labelled non-covalent contacts within the getcontacts cutoffs.
3. Count observed contacts per (prot_type, lig_type, itype) triplet.
4. Count *possible* pairs within SHELL_RADIUS (10 Å) as the normalisation
   denominator.
5. Compute contact density = observed / possible for each triplet.
6. Return a flat dict ready to be appended to the feature DataFrame.

Dependencies
------------
- getcontacts must be installed and ``get_static_contacts.py`` must be on PATH
  (or its full path supplied via GETCONTACTS_PATH env variable).
- scipy for efficient pairwise distance computation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from vlsbn.constants import INTERACTION_TYPES, SHELL_RADIUS
from vlsbn.pipeline.atomtypes import LIGAND_ATOM_TYPES, PROTEIN_ATOM_TYPES
from vlsbn.pipeline.parse import AtomRecord, ParsedComplex

logger = logging.getLogger(__name__)

# Path to getcontacts' get_static_contacts.py script.
# Override via environment variable if not on PATH.
GETCONTACTS_SCRIPT = os.environ.get(
    "GETCONTACTS_PATH", "get_static_contacts.py"
)

# Python interpreter used to run getcontacts. Defaults to the current
# interpreter but can be overridden to point at a separate conda env that
# has vmd-python installed (e.g. ~/anaconda3/envs/vmd-python/bin/python).
GETCONTACTS_PYTHON = os.environ.get("GETCONTACTS_PYTHON", sys.executable)

# Feature column sentinel for missing triplets
_ZERO = 0.0


# ---------------------------------------------------------------------------
# Temporary PDB writer
# ---------------------------------------------------------------------------

def _write_temp_pdb(complex_: ParsedComplex, dest: Path) -> None:
    """Write a minimal PDB file containing protein + ligand atoms."""
    with dest.open("w") as fh:
        serial = 1
        # Protein ATOM records
        for a in complex_.protein_atoms:
            x, y, z = a.coords
            elem = a.atom_type.split("_")[0] if "_" in a.atom_type else a.atom_type
            fh.write(
                f"ATOM  {serial:5d} {a.atom_name:<4s} {a.resname:<3s} {a.chain}"
                f"{a.resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {elem:>2s}\n"
            )
            serial += 1

        fh.write("TER\n")

        # Ligand HETATM records
        for a in complex_.ligand_atoms:
            x, y, z = a.coords
            elem = a.atom_type.split("_")[0] if "_" in a.atom_type else a.atom_type
            fh.write(
                f"HETATM{serial:5d} {a.atom_name:<4s} {a.resname:<3s} {a.chain}"
                f"{a.resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {elem:>2s}\n"
            )
            serial += 1

        fh.write("END\n")


# ---------------------------------------------------------------------------
# getcontacts wrapper
# ---------------------------------------------------------------------------

def _run_getcontacts(
    pdb_path: Path,
    ligand_resname: str,
    itypes: tuple[str, ...] = INTERACTION_TYPES,
) -> list[tuple[str, str, str, str]]:
    """Run get_static_contacts.py and parse its output.

    Parameters
    ----------
    pdb_path : Path
        Temporary PDB file with protein + ligand.
    ligand_resname : str
        Residue name of the ligand (used for the --sele2 argument).
    itypes : tuple[str, ...]
        Interaction types to detect.

    Returns
    -------
    list of (itype, prot_chain_resname_resid_atomname,
                     lig_chain_resname_resid_atomname)  tuples
        Raw contact records as returned by getcontacts.
    """
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [
        GETCONTACTS_PYTHON, GETCONTACTS_SCRIPT,
        "--structure",  str(pdb_path),
        "--output",     str(out_path),
        "--itypes",     *itypes,
        "--sele",       "protein",
        "--sele2",      f"resname {ligand_resname}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning("getcontacts error: %s", result.stderr[:500])
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("getcontacts call failed: %s", exc)
        return []

    contacts: list[tuple[str, str, str, str]] = []
    with out_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            itype = parts[1]
            if itype.startswith("hb") and len(parts) >= 5:
                # hb* format: frame  hbbb/hbss/hbsb  donor_heavy  hydrogen  acceptor_heavy
                # parts[3] is the H atom (not indexed); use donor + acceptor
                atom1, atom2 = parts[2], parts[4]
                itype = "hb"  # normalise subtypes → single "hb" feature column
            else:
                # All other itypes: frame  itype  atom1  atom2
                atom1, atom2 = parts[2], parts[3]
            contacts.append((itype, atom1, atom2))

    out_path.unlink(missing_ok=True)
    return contacts


# ---------------------------------------------------------------------------
# Atom-type index for fast lookup
# ---------------------------------------------------------------------------

def _build_type_index(
    atoms: list[AtomRecord],
) -> dict[str, list[int]]:
    """Map atom_type → list of atom indices."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(atoms):
        idx[a.atom_type].append(i)
    return dict(idx)


def _atom_name_to_index(atoms: list[AtomRecord]) -> dict[str, int]:
    """Map 'CHAIN:RESNAME:RESID:ATOMNAME' → atom index (getcontacts format)."""
    return {
        f"{a.chain}:{a.resname}:{a.resid}:{a.atom_name}": i
        for i, a in enumerate(atoms)
    }


# ---------------------------------------------------------------------------
# Shell-radius pair counting (normalisation denominator)
# ---------------------------------------------------------------------------

def _count_possible_pairs(
    protein_atoms: list[AtomRecord],
    ligand_atoms: list[AtomRecord],
    shell_radius: float = SHELL_RADIUS,
) -> dict[tuple[str, str], int]:
    """Count (prot_type, lig_type) pairs within shell_radius.

    This is the denominator for contact-density normalisation.

    Returns
    -------
    dict mapping (prot_type, lig_type) → count of pairs within shell.
    """
    if not protein_atoms or not ligand_atoms:
        return {}

    prot_coords = np.array([a.coords for a in protein_atoms], dtype=np.float32)
    lig_coords  = np.array([a.coords for a in ligand_atoms],  dtype=np.float32)

    dists = cdist(prot_coords, lig_coords, metric="euclidean")
    within = dists <= shell_radius   # boolean mask (n_prot × n_lig)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    prot_idx, lig_idx = np.where(within)
    for pi, li in zip(prot_idx, lig_idx):
        key = (protein_atoms[pi].atom_type, ligand_atoms[li].atom_type)
        counts[key] += 1

    return dict(counts)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features(
    complex_: ParsedComplex,
    shell_radius: float = SHELL_RADIUS,
    itypes: tuple[str, ...] = INTERACTION_TYPES,
) -> dict[str, float]:
    """Compute normalised contact-density features for one complex.

    Each feature key has the form ``"{prot_type}__{lig_type}__{itype}"``.
    All possible triplet combinations are included; missing ones are 0.0.

    Parameters
    ----------
    complex_ : ParsedComplex
        Parsed protein–ligand complex.
    shell_radius : float
        Outer distance cutoff for counting possible pairs (Å).
    itypes : tuple[str, ...]
        Interaction types to compute.

    Returns
    -------
    dict[str, float]
        Feature vector ready for DataFrame row insertion.
    """
    features: dict[str, float] = {}

    # --- Step 1: possible pair counts (denominator) ---
    possible = _count_possible_pairs(
        complex_.protein_atoms, complex_.ligand_atoms, shell_radius
    )

    # --- Step 2: observed contact counts via getcontacts ---
    observed: dict[tuple[str, str, str], int] = defaultdict(int)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / f"{complex_.ligand_id}.pdb"
        _write_temp_pdb(complex_, pdb_path)

        prot_name_idx = _atom_name_to_index(complex_.protein_atoms)
        lig_name_idx  = _atom_name_to_index(complex_.ligand_atoms)

        raw_contacts = _run_getcontacts(
            pdb_path, complex_.ligand_resname, itypes
        )

    for itype, atom1, atom2 in raw_contacts:
        # getcontacts lists protein atom first, ligand second
        pi = prot_name_idx.get(atom1)
        li = lig_name_idx.get(atom2)
        if pi is None or li is None:
            # Try swapped order
            pi = prot_name_idx.get(atom2)
            li = lig_name_idx.get(atom1)
        if pi is None or li is None:
            continue
        ptype = complex_.protein_atoms[pi].atom_type
        ltype = complex_.ligand_atoms[li].atom_type
        observed[(ptype, ltype, itype)] += 1

    # --- Step 3: compute density and build feature dict ---
    # Use the fixed LIGAND_ATOM_TYPES vocabulary so the feature matrix always
    # has the same columns regardless of which complexes are in the dataset.
    for ptype, ltype, itype in product(PROTEIN_ATOM_TYPES, LIGAND_ATOM_TYPES, itypes):
        key = f"{ptype}__{ltype}__{itype}"
        denom = possible.get((ptype, ltype), 0)
        if denom == 0:
            features[key] = _ZERO
        else:
            features[key] = observed.get((ptype, ltype, itype), 0) / denom

    return features


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def parse_feature_key(key: str) -> tuple[str, str, str]:
    """Split a feature key back into (prot_type, lig_type, itype)."""
    ptype, ltype, itype = key.split("__")
    return ptype, ltype, itype

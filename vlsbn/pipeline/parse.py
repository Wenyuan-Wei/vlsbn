"""
PDB structure parsing and atom-type assignment.

For each PDB file this module produces one ParsedComplex per valid ligand
found in the structure.  Protein atoms are typed via the lookup table in
atomtypes.py; ligand atoms are typed using RDKit.

Typical usage
-------------
    complexes = parse_pdb(Path("data/raw/1abc.pdb"))
    for c in complexes:
        print(c.pdb_id, c.ligand_resname, len(c.protein_atoms), len(c.ligand_atoms))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from Bio import PDB
from rdkit import Chem
from rdkit.Chem import AllChem

from vlsbn.constants import ARTIFACT_LIGANDS, MIN_LIGAND_HEAVY_ATOMS
from vlsbn.pipeline.atomtypes import get_ligand_atom_type, get_protein_atom_type

logger = logging.getLogger(__name__)

# Suppress noisy BioPython PDB warnings (missing atoms, SEQRES mismatches, etc.)
import warnings
warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AtomRecord:
    """Single heavy atom with its type label and Cartesian coordinates."""
    chain:     str
    resname:   str
    resid:     int
    atom_name: str
    atom_type: str          # vocabulary-defined type string
    coords:    np.ndarray   # shape (3,), dtype float32


@dataclass
class ParsedComplex:
    """One protein–ligand complex extracted from a PDB structure.

    A single PDB file may yield multiple ParsedComplex objects when it
    contains more than one non-artifact ligand.
    """
    pdb_id:         str
    ligand_id:      str              # unique identifier: "RESNAME_CHAIN_RESSEQ"
    ligand_resname: str
    protein_atoms:  list[AtomRecord] = field(default_factory=list)
    ligand_atoms:   list[AtomRecord] = field(default_factory=list)

    @property
    def n_protein_atoms(self) -> int:
        return len(self.protein_atoms)

    @property
    def n_ligand_atoms(self) -> int:
        return len(self.ligand_atoms)


# ---------------------------------------------------------------------------
# Standard amino-acid residue names
# ---------------------------------------------------------------------------

_STANDARD_RESIDUES: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    # Common non-standard that map cleanly to standard chemistry
    "MSE",   # selenomethionine → treat as MET
})


# ---------------------------------------------------------------------------
# Ligand atom typing via RDKit (SMILES/SDF from PDB HETATM)
# ---------------------------------------------------------------------------

def _type_ligand_atoms(
    residue: PDB.Residue.Residue,
) -> dict[str, str]:
    """Attempt to assign RDKit-based atom types to a ligand residue.

    Falls back to element-only labels if RDKit cannot parse the residue.

    Returns
    -------
    dict[str, str]
        Mapping atom_name → atom_type string.
    """
    # Build an RDKit Mol from the HETATM block using element + connectivity.
    # We write a minimal PDB block for just this residue and let RDKit parse it.
    pdb_lines = ["REMARK  ligand\n"]
    for atom in residue.get_atoms():
        element = atom.element.strip() if atom.element else "C"
        x, y, z = atom.get_vector().get_array()
        name = atom.get_name().strip()
        pdb_lines.append(
            f"HETATM{1:5d} {name:<4s} {residue.get_resname():<3s} A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
        )
    pdb_lines.append("END\n")
    pdb_block = "".join(pdb_lines)

    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=True)

    type_map: dict[str, str] = {}
    if mol is None:
        # RDKit failed — fall back to element only
        for atom in residue.get_atoms():
            elem = (atom.element or "C").strip().capitalize()
            type_map[atom.get_name().strip()] = elem
        return type_map

    atom_names = [a.get_name().strip() for a in residue.get_atoms()]
    for i, rdatom in enumerate(mol.GetAtoms()):
        if i < len(atom_names):
            type_map[atom_names[i]] = get_ligand_atom_type(rdatom)

    return type_map


# ---------------------------------------------------------------------------
# Main parsing entry point
# ---------------------------------------------------------------------------

def parse_pdb(pdb_path: Path) -> list[ParsedComplex]:
    """Parse a PDB file and return one ParsedComplex per valid ligand.

    Parameters
    ----------
    pdb_path : Path
        Path to a .pdb file.

    Returns
    -------
    list[ParsedComplex]
        Empty list if no valid ligands are found or the file cannot be parsed.
    """
    pdb_id = pdb_path.stem.upper()
    parser = PDB.PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as exc:
        logger.warning("Cannot parse %s: %s", pdb_path.name, exc)
        return []

    # Use the first model only (handles NMR ensembles gracefully)
    model = next(iter(structure))

    # ------------------------------------------------------------------
    # Collect protein atoms
    # ------------------------------------------------------------------
    protein_atoms: list[AtomRecord] = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip()
            if resname == "MSE":
                resname = "MET"   # treat selenomethionine as methionine
            if resname not in _STANDARD_RESIDUES:
                continue
            resid = residue.get_id()[1]
            for atom in residue.get_atoms():
                if atom.element == "H":
                    continue    # skip hydrogens (should be absent in X-ray)
                atom_name = atom.get_name().strip()
                atype = get_protein_atom_type(resname, atom_name)
                protein_atoms.append(AtomRecord(
                    chain=chain.get_id(),
                    resname=resname,
                    resid=resid,
                    atom_name=atom_name,
                    atom_type=atype,
                    coords=atom.get_vector().get_array().astype(np.float32),
                ))

    if not protein_atoms:
        logger.debug("%s: no protein atoms found; skipping.", pdb_id)
        return []

    # ------------------------------------------------------------------
    # Identify valid ligands and build one ParsedComplex per ligand
    # ------------------------------------------------------------------
    complexes: list[ParsedComplex] = []

    for chain in model:
        for residue in chain:
            het_flag, resseq, _ = residue.get_id()
            resname = residue.get_resname().strip()

            # HETATM residues have a non-blank het_flag ("H_" prefix)
            if not het_flag.strip():
                continue
            if resname in ("HOH", "WAT"):
                continue
            if resname in ARTIFACT_LIGANDS:
                continue

            # Reject tiny molecules (ions, solvents) by heavy-atom count
            n_heavy = sum(
                1 for a in residue.get_atoms()
                if (a.element or "").strip().upper() != "H"
            )
            if n_heavy < MIN_LIGAND_HEAVY_ATOMS:
                continue

            ligand_id = f"{resname}_{chain.get_id()}_{resseq}"
            type_map = _type_ligand_atoms(residue)

            ligand_atoms: list[AtomRecord] = []
            for atom in residue.get_atoms():
                if atom.element == "H":
                    continue
                atom_name = atom.get_name().strip()
                ligand_atoms.append(AtomRecord(
                    chain=chain.get_id(),
                    resname=resname,
                    resid=resseq,
                    atom_name=atom_name,
                    atom_type=type_map.get(atom_name, "OTHER"),
                    coords=atom.get_vector().get_array().astype(np.float32),
                ))

            if not ligand_atoms:
                continue

            complexes.append(ParsedComplex(
                pdb_id=pdb_id,
                ligand_id=ligand_id,
                ligand_resname=resname,
                protein_atoms=protein_atoms,   # shared reference — read-only
                ligand_atoms=ligand_atoms,
            ))

    logger.debug("%s: %d valid ligand(s) found.", pdb_id, len(complexes))
    return complexes

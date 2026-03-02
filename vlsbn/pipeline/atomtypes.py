"""
Atom-type taxonomy for proteins and ligands.

Protein types are assigned via a (residue_name, atom_name) lookup table built
from standard amino-acid chemistry.  Ligand types are assigned at runtime
using RDKit atom properties (element, hybridisation, aromaticity, charge).

Protein atom-type vocabulary
----------------------------
C_ali   aliphatic carbon   (CA, CB, non-aromatic/non-carbonyl sidechain C)
C_aro   aromatic carbon    (PHE/TYR/TRP/HIS ring carbons)
C_car   carbonyl carbon    (backbone C=O, Asn/Gln/Asp/Glu sidechain C=O)
N_bb    backbone amide N
N_ami   sidechain amide N  (Asn ND2, Gln NE2)
N_aro   aromatic N         (His ND1/NE2, Trp NE1)
N_pos   positively charged N (Arg NH1/NH2/NE, Lys NZ)
O_bb    backbone carbonyl O
O_hyd   hydroxyl O         (Ser OG, Thr OG1, Tyr OH)
O_car   uncharged sidechain carbonyl O (Asn OD1, Gln OE1)
O_neg   negatively charged O (Asp OD1/OD2, Glu OE1/OE2, C-term OXT)
S_sul   sulfur             (Cys SG, Met SD)
OTHER   any atom not covered by the above (metals, phosphorus, halogens, etc.)

Ligand atom-type vocabulary (generated dynamically)
----------------------------------------------------
Format: {element}_{hybridisation}[_aro][_pos|_neg]
Examples: C_3, C_2, C_aro, N_3_pos, O_2, S_3, F, Cl, Br, I, P_3, X (other)
"""

from __future__ import annotations

from rdkit.Chem import rdchem

# ---------------------------------------------------------------------------
# Protein atom typing — lookup tables
# ---------------------------------------------------------------------------

PROTEIN_ATOM_TYPES: tuple[str, ...] = (
    "C_ali", "C_aro", "C_car",
    "N_bb",  "N_ami", "N_aro", "N_pos",
    "O_bb",  "O_hyd", "O_car", "O_neg",
    "S_sul",
    "OTHER",
)

# Backbone atoms shared by all standard residues
_BACKBONE: dict[str, str] = {
    "N":   "N_bb",
    "CA":  "C_ali",
    "C":   "C_car",
    "O":   "O_bb",
    "OXT": "O_neg",  # C-terminal carboxylate oxygen
}

# Sidechain atoms per residue
_SIDECHAIN: dict[str, dict[str, str]] = {
    "ALA": {"CB": "C_ali"},
    "ARG": {
        "CB": "C_ali", "CG": "C_ali", "CD": "C_ali",
        "NE": "N_pos", "CZ": "C_ali", "NH1": "N_pos", "NH2": "N_pos",
    },
    "ASN": {"CB": "C_ali", "CG": "C_car", "OD1": "O_car", "ND2": "N_ami"},
    "ASP": {"CB": "C_ali", "CG": "C_car", "OD1": "O_neg", "OD2": "O_neg"},
    "CYS": {"CB": "C_ali", "SG": "S_sul"},
    "GLN": {
        "CB": "C_ali", "CG": "C_ali",
        "CD": "C_car", "OE1": "O_car", "NE2": "N_ami",
    },
    "GLU": {
        "CB": "C_ali", "CG": "C_ali",
        "CD": "C_car", "OE1": "O_neg", "OE2": "O_neg",
    },
    "GLY": {},
    "HIS": {
        "CB": "C_ali",
        "CG": "C_aro", "ND1": "N_aro", "CD2": "C_aro",
        "CE1": "C_aro", "NE2": "N_aro",
    },
    "ILE": {"CB": "C_ali", "CG1": "C_ali", "CG2": "C_ali", "CD1": "C_ali"},
    "LEU": {"CB": "C_ali", "CG": "C_ali", "CD1": "C_ali", "CD2": "C_ali"},
    "LYS": {
        "CB": "C_ali", "CG": "C_ali", "CD": "C_ali",
        "CE": "C_ali", "NZ": "N_pos",
    },
    "MET": {"CB": "C_ali", "CG": "C_ali", "SD": "S_sul", "CE": "C_ali"},
    "PHE": {
        "CB": "C_ali",
        "CG": "C_aro", "CD1": "C_aro", "CD2": "C_aro",
        "CE1": "C_aro", "CE2": "C_aro", "CZ": "C_aro",
    },
    "PRO": {"CB": "C_ali", "CG": "C_ali", "CD": "C_ali"},
    "SER": {"CB": "C_ali", "OG": "O_hyd"},
    "THR": {"CB": "C_ali", "OG1": "O_hyd", "CG2": "C_ali"},
    "TRP": {
        "CB": "C_ali",
        "CG": "C_aro", "CD1": "C_aro", "CD2": "C_aro",
        "NE1": "N_aro", "CE2": "C_aro", "CE3": "C_aro",
        "CZ2": "C_aro", "CZ3": "C_aro", "CH2": "C_aro",
    },
    "TYR": {
        "CB": "C_ali",
        "CG": "C_aro", "CD1": "C_aro", "CD2": "C_aro",
        "CE1": "C_aro", "CE2": "C_aro", "CZ": "C_aro",
        "OH": "O_hyd",
    },
    "VAL": {"CB": "C_ali", "CG1": "C_ali", "CG2": "C_ali"},
}

# Flatten backbone + sidechains into a single lookup:
#   (resname, atom_name) -> type  (backbone uses "*" as wildcard resname)
PROTEIN_TYPE_LOOKUP: dict[tuple[str, str], str] = {}
for _atom, _type in _BACKBONE.items():
    PROTEIN_TYPE_LOOKUP[("*", _atom)] = _type
for _res, _atoms in _SIDECHAIN.items():
    for _atom, _type in _atoms.items():
        PROTEIN_TYPE_LOOKUP[(_res, _atom)] = _type


def get_protein_atom_type(resname: str, atom_name: str) -> str:
    """Return the protein atom-type string for a given residue / atom pair.

    Falls back to "OTHER" for atoms not in the lookup (e.g. metals,
    non-standard residues, or alternate conformations with unusual names).
    """
    # Strip alternate-location suffixes (e.g. "CA" vs "CA ")
    atom_name = atom_name.strip()
    return (
        PROTEIN_TYPE_LOOKUP.get(("*", atom_name))
        or PROTEIN_TYPE_LOOKUP.get((resname, atom_name))
        or "OTHER"
    )


# ---------------------------------------------------------------------------
# Ligand atom typing — RDKit-based
# ---------------------------------------------------------------------------

_HYBRID_LABEL: dict[rdchem.HybridizationType, str] = {
    rdchem.HybridizationType.SP3: "3",
    rdchem.HybridizationType.SP2: "2",
    rdchem.HybridizationType.SP:  "1",
}

# Elements for which we encode hybridisation + charge
_TYPED_ELEMENTS: frozenset[int] = frozenset({6, 7, 8, 16, 15})  # C N O S P

# Elements that are just reported as their symbol
_HALOGEN_NUMS: dict[int, str] = {9: "F", 17: "Cl", 35: "Br", 53: "I"}

_ELEMENT_SYMBOL: dict[int, str] = {
    6: "C", 7: "N", 8: "O", 16: "S", 15: "P",
}


def get_ligand_atom_type(atom: rdchem.Atom) -> str:
    """Return the ligand atom-type string for an RDKit atom.

    Type format: {element}_{hybridisation}[_aro][_pos|_neg]
    Halogens   : element symbol only (F, Cl, Br, I)
    Other      : "X{atomic_num}" (metals, unusual elements)
    """
    anum = atom.GetAtomicNum()

    if anum in _HALOGEN_NUMS:
        return _HALOGEN_NUMS[anum]

    if anum not in _TYPED_ELEMENTS:
        return f"X{anum}"

    elem = _ELEMENT_SYMBOL[anum]

    if atom.GetIsAromatic():
        return f"{elem}_aro"

    hybrid = _HYBRID_LABEL.get(atom.GetHybridization(), "x")
    charge = atom.GetFormalCharge()
    suffix = "_pos" if charge > 0 else ("_neg" if charge < 0 else "")
    return f"{elem}_{hybrid}{suffix}"

"""
Unit tests for the pipeline subpackage.

Run with:  pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest

from vlsbn.pipeline.atomtypes import (
    get_ligand_atom_type,
    get_protein_atom_type,
)
from vlsbn.pipeline.parse import ParsedComplex, AtomRecord


# ---------------------------------------------------------------------------
# Protein atom typing
# ---------------------------------------------------------------------------

class TestProteinAtomTyping:
    def test_backbone_N(self):
        assert get_protein_atom_type("ALA", "N") == "N_bb"

    def test_backbone_CA(self):
        assert get_protein_atom_type("GLY", "CA") == "C_ali"

    def test_backbone_carbonyl(self):
        assert get_protein_atom_type("VAL", "C") == "C_car"

    def test_backbone_O(self):
        assert get_protein_atom_type("LEU", "O") == "O_bb"

    def test_his_aromatic_carbon(self):
        assert get_protein_atom_type("HIS", "CE1") == "C_aro"

    def test_arg_charged_N(self):
        assert get_protein_atom_type("ARG", "NH1") == "N_pos"

    def test_asp_neg_O(self):
        assert get_protein_atom_type("ASP", "OD1") == "O_neg"

    def test_cys_sulfur(self):
        assert get_protein_atom_type("CYS", "SG") == "S_sul"

    def test_tyr_hydroxyl(self):
        assert get_protein_atom_type("TYR", "OH") == "O_hyd"

    def test_unknown_atom(self):
        assert get_protein_atom_type("ALA", "XX") == "OTHER"

    def test_unknown_residue(self):
        # Non-standard residue with a backbone-named atom
        assert get_protein_atom_type("XYZ", "N") == "N_bb"


# ---------------------------------------------------------------------------
# Ligand atom typing (RDKit-based)
# ---------------------------------------------------------------------------

class TestLigandAtomTyping:
    """Construct minimal RDKit atoms for testing."""

    def _make_atom(self, atomic_num, hybridization, aromatic=False, charge=0):
        from rdkit.Chem import RWMol, Atom
        from rdkit.Chem.rdchem import HybridizationType
        mol = RWMol()
        idx = mol.AddAtom(Atom(atomic_num))
        mol.GetAtomWithIdx(idx).SetIsAromatic(aromatic)
        mol.GetAtomWithIdx(idx).SetFormalCharge(charge)
        mol.GetAtomWithIdx(idx).SetHybridization(hybridization)
        return mol.GetAtomWithIdx(idx)

    def test_sp3_carbon(self):
        from rdkit.Chem.rdchem import HybridizationType
        atom = self._make_atom(6, HybridizationType.SP3)
        assert get_ligand_atom_type(atom) == "C_3"

    def test_aromatic_carbon(self):
        from rdkit.Chem.rdchem import HybridizationType
        atom = self._make_atom(6, HybridizationType.SP2, aromatic=True)
        assert get_ligand_atom_type(atom) == "C_aro"

    def test_sp2_nitrogen(self):
        from rdkit.Chem.rdchem import HybridizationType
        atom = self._make_atom(7, HybridizationType.SP2)
        assert get_ligand_atom_type(atom) == "N_2"

    def test_positive_nitrogen(self):
        from rdkit.Chem.rdchem import HybridizationType
        atom = self._make_atom(7, HybridizationType.SP3, charge=1)
        assert get_ligand_atom_type(atom) == "N_3_pos"

    def test_fluorine(self):
        from rdkit.Chem.rdchem import HybridizationType
        atom = self._make_atom(9, HybridizationType.SP3)
        assert get_ligand_atom_type(atom) == "F"

    def test_chlorine(self):
        from rdkit.Chem.rdchem import HybridizationType
        atom = self._make_atom(17, HybridizationType.SP3)
        assert get_ligand_atom_type(atom) == "Cl"

    def test_metal(self):
        from rdkit.Chem.rdchem import HybridizationType
        atom = self._make_atom(30, HybridizationType.UNSPECIFIED)  # Zn
        assert get_ligand_atom_type(atom) == "X30"


# ---------------------------------------------------------------------------
# ParsedComplex helpers
# ---------------------------------------------------------------------------

def _make_atom_record(atype: str, coords=(0.0, 0.0, 0.0)) -> AtomRecord:
    return AtomRecord(
        chain="A", resname="ALA", resid=1,
        atom_name="CA", atom_type=atype,
        coords=np.array(coords, dtype=np.float32),
    )


class TestParsedComplex:
    def test_n_protein_atoms(self):
        c = ParsedComplex(
            pdb_id="TEST", ligand_id="LIG_A_1", ligand_resname="LIG",
            protein_atoms=[_make_atom_record("C_ali")] * 5,
            ligand_atoms=[_make_atom_record("C_3")] * 2,
        )
        assert c.n_protein_atoms == 5
        assert c.n_ligand_atoms == 2

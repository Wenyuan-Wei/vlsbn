"""
Project-wide constants for VLS_BN.

All magic numbers live here so every module reads from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"        # downloaded PDB files
FILTERED_DIR  = DATA_DIR / "filtered"   # post-quality-filter symlinks / copies
PROCESSED_DIR = DATA_DIR / "processed"  # per-complex feature parquets

# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

# Generous outer shell used as the denominator for contact-density normalisation.
# Any atom pair whose heavy-atom distance is within this radius is counted as
# a "possible" interaction regardless of interaction type.
SHELL_RADIUS = 10.0  # Å

# Interaction type labels (sourced from getcontacts argparsers.py)
INTERACTION_TYPES: tuple[str, ...] = ("hb", "sb", "pc", "ps", "ts", "vdw")

# Distance cutoffs per interaction type (used by getcontacts internally).
# Kept here for reference and for any custom re-implementation.
CONTACT_CUTOFFS: dict[str, float] = {
    "hb": 3.5,   # hydrogen bond  donor–acceptor distance
    "sb": 4.0,   # salt bridge    charge–charge distance
    "ts": 5.0,   # T-stacking     ring centroid distance
    "pc": 6.0,   # pi–cation      ring centroid–cation distance
    "ps": 7.0,   # pi–stacking    ring centroid distance
    # vdw is pair-specific (sum of VDW radii + 0.5 Å); handled at runtime
}

# ---------------------------------------------------------------------------
# BN node discretisation
# ---------------------------------------------------------------------------

# Contact density is discretised into N_STATES ordered states:
#   0 → no contact   (density == 0)
#   1 → sparse       (0 < density ≤ Q1 across training set)
#   2 → moderate     (Q1 < density ≤ Q3)
#   3 → dense        (density > Q3)
N_STATES = 4

# ---------------------------------------------------------------------------
# PDB quality filters
# ---------------------------------------------------------------------------

MAX_RESOLUTION = 2.5   # Å  — X-ray resolution upper bound
MAX_RFREE      = 0.35  # Rₓ — R-free upper bound

# Minimum number of heavy atoms a HETATM residue must have to be considered
# a drug-like ligand.  Ions, solvents, and tiny fragments are rejected below
# this threshold before the artifact-name list is even consulted.
MIN_LIGAND_HEAVY_ATOMS: int = 6

# HETATM residue codes excluded as crystallographic artifacts / solvent / ions.
ARTIFACT_LIGANDS: frozenset[str] = frozenset({
    # --- Standard amino acids appearing as HETATM (free AA in solution, etc.) ---
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    # D-amino acids
    "DAL", "DAR", "DSG", "DAS", "DCY", "DGL", "DGN", "DGL",
    "DHI", "DIL", "DLE", "DLY", "MED", "DPN", "DPR", "DSN",
    "DTH", "DTR", "DTY", "DVA",

    # --- Solvents & cryoprotectants ---
    "GOL", "EDO", "PEG", "EOH", "MPD", "DMS", "ACE", "ACT", "FMT",
    "HEZ", "PGE", "BOG", "ETA", "MOH", "IPA", "ACN", "TFE", "BTN",
    "PGO", "DOD", "THF", "DCM", "DIO", "CCN", "DMF", "P33", "PE4",
    "PE5", "PE6", "PE7", "PE8", "P6G", "PG4",

    # --- Buffers & crystallisation additives ---
    "EPE", "MES", "TRS", "NH4", "IMD", "BIS", "CAPS", "CHES",
    "PIPES", "TMP", "GAI", "TAM", "TCN",

    # --- Inorganic ions (single-atom or simple polyatomic) ---
    "CL", "NA", "MG", "ZN", "CA", "K", "MN", "FE", "CU", "CO",
    "NI", "CD", "IOD", "BR", "SO4", "PO4", "AZI", "CS", "RB",
    "SR", "BA", "AU", "AG", "PT", "HG", "PB", "NO3", "NO2",
    "SCN", "CLO", "SMO", "VO4", "ACY", "FLC", "LI",

    # --- Modified / non-standard amino-acid residues ---
    "MSE", "SEP", "TPO", "PTR", "MLY", "OCS", "CME", "CSD",
    "CSX", "YCM", "KCX", "ALY", "LLP", "PLP", "SEC", "CGU",
    "HYP", "FME", "MVA", "DIV",

    # --- Glycans & sugars (glycosylation on protein surface) ---
    "NAG", "BMA", "MAN", "FUC", "GAL", "BGC", "XYP", "NDG",
    "NAE", "SIA", "FCA", "A2G", "LAT", "RIB", "GLA", "GLC",
    "GXL", "GNS", "IDR", "SHB", "LFR", "AFL",

    # --- Lipids / fatty acids used as crystallisation agents ---
    "OLC", "PLM", "STE", "OLA", "LDA", "LPP", "LMT", "LMU",
    "LMR", "PGV",

    # --- Catch-alls ---
    "UNK", "UNL",
})

# ---------------------------------------------------------------------------
# RCSB endpoints
# ---------------------------------------------------------------------------

RCSB_SEARCH_URL    = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL  = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

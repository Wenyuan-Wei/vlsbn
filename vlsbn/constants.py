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

# HETATM residue codes excluded as crystallographic artifacts / solvent / ions.
# Extend this list as needed during data exploration.
ARTIFACT_LIGANDS: frozenset[str] = frozenset({
    # Solvents & cryoprotectants
    "GOL", "EDO", "PEG", "EOH", "MPD", "DMS", "ACE", "ACT", "FMT",
    "HEZ", "PGE", "BOG",
    # Buffers
    "EPE", "MES", "TRS", "NH4", "IMD",
    # Inorganic ions
    "CL", "NA", "MG", "ZN", "CA", "K", "MN", "FE", "CU", "CO",
    "NI", "CD", "IOD", "BR", "SO4", "PO4", "AZI",
    # Modified amino-acid residues (non-standard but not drug-like)
    "MSE", "SEP", "TPO", "PTR", "MLY", "OCS", "CME", "CSD",
    "CSX", "YCM", "KCX", "ALY", "LLP", "PLP", "SEC",
    # Catch-alls
    "UNK", "UNL",
})

# ---------------------------------------------------------------------------
# RCSB endpoints
# ---------------------------------------------------------------------------

RCSB_SEARCH_URL   = "https://search.rcsb.org/rcsbsearch/v1/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

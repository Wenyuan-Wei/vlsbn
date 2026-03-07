"""
RCSB PDB download and quality filtering.

Typical usage
-------------
    pdb_ids = fetch_pdb_ids()
    paths   = download_all(pdb_ids, dest_dir=RAW_DIR)
"""

from __future__ import annotations

import io
import logging
import tempfile
import time
from pathlib import Path

import requests

from vlsbn.constants import (
    ARTIFACT_LIGANDS,
    MAX_RESOLUTION,
    MAX_RFREE,
    RCSB_CIF_DOWNLOAD_URL,
    RCSB_DOWNLOAD_URL,
    RCSB_SEARCH_URL,
    RAW_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RCSB Search API
# ---------------------------------------------------------------------------

def _build_query(max_resolution: float, max_rfree: float) -> dict:
    """Construct an RCSB Search API v2 JSON query.

    Filters applied server-side:
    - X-ray crystallography only
    - Resolution ≤ max_resolution Å
    - R-free ≤ max_rfree
    - ≥ 1 protein chain
    - ≥ 1 non-polymer (ligand) entity
    - 0 nucleic acid chains  (excludes DNA/RNA complexes)
    """
    def _text_node(attribute: str, operator: str, value) -> dict:
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": attribute,
                "operator": operator,
                "value": value,
            },
        }

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                _text_node(
                    "exptl.method",
                    "exact_match",
                    "X-RAY DIFFRACTION",
                ),
                _text_node(
                    "rcsb_entry_info.resolution_combined",
                    "less_or_equal",
                    max_resolution,
                ),
                _text_node(
                    "refine.ls_R_factor_R_free",
                    "less_or_equal",
                    max_rfree,
                ),
                _text_node(
                    "rcsb_entry_info.polymer_entity_count_protein",
                    "greater_or_equal",
                    1,
                ),
                _text_node(
                    "rcsb_entry_info.nonpolymer_entity_count",
                    "greater_or_equal",
                    1,
                ),
                _text_node(
                    "rcsb_entry_info.polymer_entity_count_nucleic_acid",
                    "equals",
                    0,
                ),
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }


def fetch_pdb_ids(
    max_resolution: float = MAX_RESOLUTION,
    max_rfree: float = MAX_RFREE,
) -> list[str]:
    """Query RCSB and return a list of PDB IDs matching quality filters.

    Parameters
    ----------
    max_resolution : float
        Upper bound on X-ray resolution (Å).
    max_rfree : float
        Upper bound on R-free value.

    Returns
    -------
    list[str]
        Uppercase PDB IDs (e.g. ["1ABC", "2XYZ", ...]).
    """
    query = _build_query(max_resolution, max_rfree)
    logger.info(
        "Querying RCSB (resolution ≤ %.1f Å, R-free ≤ %.2f)…",
        max_resolution,
        max_rfree,
    )
    response = requests.post(
        RCSB_SEARCH_URL,
        json=query,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()
    hits = response.json().get("result_set", [])
    ids = [h["identifier"].upper() for h in hits]
    logger.info("Found %d PDB entries.", len(ids))
    return ids


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _cif_to_pdb_bytes(pdb_id: str, cif_bytes: bytes) -> bytes:
    """Convert mmCIF bytes to legacy PDB-format bytes via BioPython.

    Used as a fallback for entries whose .pdb file is no longer served by
    RCSB (typically structures deposited after the PDB format deprecation).
    BioPython's MMCIFParser → PDBIO round-trip preserves ATOM/HETATM records
    consistently with the rest of the pipeline's PDBParser.
    """
    from Bio import PDB

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(cif_bytes)
        tmp_path = Path(tmp.name)

    try:
        parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(tmp_path))
        buf = io.StringIO()
        pdb_io = PDB.PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(buf)
        return buf.getvalue().encode("utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)


def download_pdb(
    pdb_id: str,
    dest_dir: Path = RAW_DIR,
    delay: float = 0.05,
) -> Path:
    """Download a single PDB file from RCSB, with CIF fallback.

    Tries the legacy .pdb URL first.  If RCSB returns 404 (common for
    entries deposited after ~2024 that only exist in mmCIF format), falls
    back to downloading the .cif file and converting it to PDB format via
    BioPython before saving.  Skips download if the file already exists.

    Parameters
    ----------
    pdb_id : str
        PDB ID (case-insensitive).
    dest_dir : Path
        Directory to save the file.
    delay : float
        Seconds to sleep after each successful download (rate-limiting).

    Returns
    -------
    Path
        Path to the downloaded (or pre-existing) .pdb file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{pdb_id.lower()}.pdb"
    if dest.exists():
        return dest

    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    resp = requests.get(url, timeout=60)

    if resp.status_code == 404:
        logger.debug("%s: .pdb not found, trying .cif fallback.", pdb_id)
        cif_url = RCSB_CIF_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
        resp = requests.get(cif_url, timeout=60)
        resp.raise_for_status()
        pdb_bytes = _cif_to_pdb_bytes(pdb_id, resp.content)
        dest.write_bytes(pdb_bytes)
        logger.debug("%s: converted from CIF → %s", pdb_id, dest)
    else:
        resp.raise_for_status()
        dest.write_bytes(resp.content)

    time.sleep(delay)
    return dest


def download_all(
    pdb_ids: list[str],
    dest_dir: Path = RAW_DIR,
    delay: float = 0.05,
    log_every: int = 500,
) -> list[Path]:
    """Download all PDB files, skipping those already present.

    Parameters
    ----------
    pdb_ids : list[str]
        List of PDB IDs to download.
    dest_dir : Path
        Destination directory.
    delay : float
        Per-request sleep to avoid hammering RCSB.
    log_every : int
        Log progress every N structures.

    Returns
    -------
    list[Path]
        Paths to successfully downloaded files.
    """
    paths: list[Path] = []
    failed: list[str] = []

    for i, pdb_id in enumerate(pdb_ids, start=1):
        try:
            path = download_pdb(pdb_id, dest_dir, delay)
            paths.append(path)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", pdb_id, exc)
            failed.append(pdb_id)

        if i % log_every == 0:
            logger.info("Progress: %d / %d  (%d failed)", i, len(pdb_ids), len(failed))

    logger.info(
        "Download complete: %d succeeded, %d failed.", len(paths), len(failed)
    )
    return paths


# ---------------------------------------------------------------------------
# Lightweight post-download filtering
# ---------------------------------------------------------------------------

def has_valid_ligand(pdb_path: Path) -> bool:
    """Return True if the PDB file contains at least one non-artifact HETATM.

    This is a fast text-scan fallback; the main quality filters are applied
    server-side via the RCSB query.  Use this to catch edge cases (e.g. an
    entry that passed the API filter but whose only HETATM is water or a
    buffer ion).
    """
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname not in ("HOH", "WAT") and resname not in ARTIFACT_LIGANDS:
                return True
    return False

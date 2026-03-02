"""
RCSB PDB download and quality filtering.

Typical usage
-------------
    pdb_ids = fetch_pdb_ids()
    paths   = download_all(pdb_ids, dest_dir=RAW_DIR)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from vlsbn.constants import (
    ARTIFACT_LIGANDS,
    MAX_RESOLUTION,
    MAX_RFREE,
    RCSB_DOWNLOAD_URL,
    RCSB_SEARCH_URL,
    RAW_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RCSB Search API
# ---------------------------------------------------------------------------

def _build_query(max_resolution: float, max_rfree: float) -> dict:
    """Construct an RCSB Search API v1 JSON query.

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
                    "rcsb_entry_info.experimental_method",
                    "exact_match",
                    "X-RAY DIFFRACTION",
                ),
                _text_node(
                    "rcsb_entry_info.resolution_combined",
                    "less_or_equal",
                    max_resolution,
                ),
                _text_node(
                    "refine.ls_rfactor_rfree",
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

def download_pdb(
    pdb_id: str,
    dest_dir: Path = RAW_DIR,
    delay: float = 0.05,
) -> Path:
    """Download a single PDB file from RCSB.

    Skips download if the file already exists locally.

    Parameters
    ----------
    pdb_id : str
        Four-character PDB ID (case-insensitive).
    dest_dir : Path
        Directory to save the file.
    delay : float
        Seconds to sleep after each successful download (rate-limiting).

    Returns
    -------
    Path
        Path to the downloaded (or pre-existing) file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{pdb_id.lower()}.pdb"
    if dest.exists():
        return dest

    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    resp = requests.get(url, timeout=60)
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

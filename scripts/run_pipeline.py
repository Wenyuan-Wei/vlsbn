#!/usr/bin/env python3
"""
End-to-end data pipeline: RCSB download → parse → contact features → parquet.

Usage
-----
    # Full run
    python scripts/run_pipeline.py

    # Quick test on first 100 structures
    python scripts/run_pipeline.py --limit 100

    # Skip download (reprocess existing PDB files)
    python scripts/run_pipeline.py --skip-download
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlsbn.constants import PROCESSED_DIR, RAW_DIR
from vlsbn.pipeline.contacts import compute_features
from vlsbn.pipeline.fetch import download_all, fetch_pdb_ids
from vlsbn.pipeline.parse import parse_pdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLS_BN data pipeline")
    p.add_argument("--max-resolution", type=float, default=2.5,
                   help="Max X-ray resolution in Å (default: 2.5)")
    p.add_argument("--max-rfree", type=float, default=0.35,
                   help="Max R-free value (default: 0.35)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N structures (for testing)")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip RCSB download; reuse files already in data/raw/")
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "contact_features.parquet",
                   help="Output parquet path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1 — Fetch & download
    # ------------------------------------------------------------------
    if args.skip_download:
        pdb_paths = sorted(RAW_DIR.glob("*.pdb"))
        logger.info("Skipping download; found %d PDB files in %s.", len(pdb_paths), RAW_DIR)
    else:
        logger.info("Fetching PDB IDs from RCSB…")
        pdb_ids = fetch_pdb_ids(args.max_resolution, args.max_rfree)
        if args.limit:
            pdb_ids = pdb_ids[: args.limit]
            logger.info("Limiting to first %d IDs.", args.limit)
        pdb_paths = download_all(pdb_ids)

    if args.limit and not args.skip_download:
        pdb_paths = pdb_paths[: args.limit]

    # ------------------------------------------------------------------
    # Step 2 — Parse + compute contact features
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    n_failed = 0

    for pdb_path in pdb_paths:
        try:
            complexes = parse_pdb(pdb_path)
        except Exception as exc:
            logger.warning("parse_pdb failed for %s: %s", pdb_path.name, exc)
            n_failed += 1
            continue

        for c in complexes:
            try:
                features = compute_features(c)
                features["pdb_id"]    = c.pdb_id
                features["ligand_id"] = c.ligand_id
                all_rows.append(features)
            except Exception as exc:
                logger.warning("compute_features failed for %s/%s: %s",
                               c.pdb_id, c.ligand_id, exc)
                n_failed += 1

    logger.info(
        "Processed %d complexes from %d structures (%d errors).",
        len(all_rows), len(pdb_paths), n_failed,
    )

    if not all_rows:
        logger.error("No complexes extracted — check pipeline settings.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3 — Save
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows).fillna(0.0)

    # Move meta columns to front
    meta = [c for c in ("pdb_id", "ligand_id") if c in df.columns]
    feat = [c for c in df.columns if c not in meta]
    df = df[meta + feat]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info("Saved %d × %d feature matrix → %s", len(df), len(feat), args.out)


if __name__ == "__main__":
    main()

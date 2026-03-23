#!/usr/bin/env python3
"""
Stage 3 of the HPC pipeline: merge chunk parquets and train the BN.

Reads all chunk_*.parquet files produced by hpc_process.py, concatenates
them, strips meta columns, and calls the standard train + reference-BN
pipeline (identical to scripts/train_bn.py).

Usage
-----
    python scripts/hpc/merge_train.py
    python scripts/hpc/merge_train.py \\
        --chunks-dir  data/chunks \\
        --out-trained  data/models/bn_trained.pkl \\
        --out-reference data/models/bn_reference.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlsbn.model.bn import train
from vlsbn.model.reference import build_reference_bn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_META_COLS = {"pdb_id", "ligand_id"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge chunks and train VLS_BN")
    p.add_argument("--chunks-dir",    type=Path, default=Path("data/chunks"))
    p.add_argument("--out-features",  type=Path, default=Path("data/processed/contact_features.parquet"),
                   help="Save merged feature matrix here (optional but useful for inspection)")
    p.add_argument("--out-trained",   type=Path, default=Path("data/models/bn_trained.pkl"))
    p.add_argument("--out-reference", type=Path, default=Path("data/models/bn_reference.pkl"))
    p.add_argument("--max-parents",   type=int,   default=4)
    p.add_argument("--pseudocount",   type=float, default=0.5)
    p.add_argument("--n-shuffles",    type=int,   default=1)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Merge chunks                                                         #
    # ------------------------------------------------------------------ #
    chunk_files = sorted(args.chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        logger.error("No chunk parquets found in %s", args.chunks_dir)
        logger.error("Run hpc_process.py array job first.")
        sys.exit(1)

    logger.info("Merging %d chunk files…", len(chunk_files))
    df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    logger.info("Merged: %d complexes total.", len(df))

    # Save merged parquet (convenient for inspection / re-training locally)
    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_features, index=False)
    logger.info("Saved merged features → %s", args.out_features)

    feature_cols = [c for c in df.columns if c not in _META_COLS]
    feature_df   = df[feature_cols].copy()
    logger.info("Feature matrix (raw): %d complexes × %d nodes.", *feature_df.shape)

    # Drop constant columns — features whose value never varies across the
    # training set carry zero information and cause MI = 0 for all edges
    # touching them, which adds noise to BaNDyT's structure search.
    constant_mask = feature_df.nunique() <= 1
    n_dropped = int(constant_mask.sum())
    feature_df = feature_df.loc[:, ~constant_mask]
    logger.info(
        "Dropped %d constant column(s) → %d informative features remain.",
        n_dropped, feature_df.shape[1],
    )

    # ------------------------------------------------------------------ #
    # Train BN                                                             #
    # ------------------------------------------------------------------ #
    logger.info("Training BN…")
    trained_bn = train(feature_df, max_parents=args.max_parents, pseudocount=args.pseudocount)
    trained_bn.save(args.out_trained)

    # ------------------------------------------------------------------ #
    # Build reference BN                                                   #
    # ------------------------------------------------------------------ #
    logger.info("Building reference BN (n_shuffles=%d)…", args.n_shuffles)
    reference_bn = build_reference_bn(
        feature_df, trained_bn,
        n_shuffles=args.n_shuffles, pseudocount=args.pseudocount, seed=args.seed,
    )
    reference_bn.save(args.out_reference)

    logger.info("Done.")
    logger.info("  Trained BN   → %s", args.out_trained)
    logger.info("  Reference BN → %s", args.out_reference)


if __name__ == "__main__":
    main()

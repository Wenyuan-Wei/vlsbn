#!/usr/bin/env python3
"""
Train the VLS_BN model: BN structure learning + CPT fitting + reference BN.

Usage
-----
    python scripts/train_bn.py

    python scripts/train_bn.py \\
        --features data/processed/contact_features.parquet \\
        --out-trained  data/models/bn_trained.pkl \\
        --out-reference data/models/bn_reference.pkl \\
        --n-shuffles 3
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlsbn.model.bn import train
from vlsbn.model.reference import build_reference_bn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_FEATURES  = Path("data/processed/contact_features.parquet")
_DEFAULT_TRAINED   = Path("data/models/bn_trained.pkl")
_DEFAULT_REFERENCE = Path("data/models/bn_reference.pkl")

_META_COLS = {"pdb_id", "ligand_id"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VLS_BN Bayesian Networks")
    p.add_argument("--features",      type=Path, default=_DEFAULT_FEATURES)
    p.add_argument("--out-trained",   type=Path, default=_DEFAULT_TRAINED)
    p.add_argument("--out-reference", type=Path, default=_DEFAULT_REFERENCE)
    p.add_argument("--max-parents",   type=int,  default=4,
                   help="Max in-degree per BN node (passed to BaNDyT)")
    p.add_argument("--pseudocount",   type=float, default=0.5,
                   help="Laplace smoothing for CPT estimation")
    p.add_argument("--n-shuffles",    type=int,  default=1,
                   help="Number of shuffles to average for reference BN")
    p.add_argument("--seed",          type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.features.exists():
        logger.error("Feature file not found: %s", args.features)
        logger.error("Run scripts/run_pipeline.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    logger.info("Loading features from %s…", args.features)
    df = pd.read_parquet(args.features)

    feature_cols = [c for c in df.columns if c not in _META_COLS]
    feature_df   = df[feature_cols].copy()
    logger.info("Feature matrix: %d complexes × %d nodes.", *feature_df.shape)

    # ------------------------------------------------------------------
    # Train BN on real data
    # ------------------------------------------------------------------
    logger.info("Training BN on real data…")
    trained_bn = train(
        feature_df,
        max_parents=args.max_parents,
        pseudocount=args.pseudocount,
    )
    trained_bn.save(args.out_trained)

    # ------------------------------------------------------------------
    # Build reference BN on shuffled pairings
    # ------------------------------------------------------------------
    logger.info("Building reference BN (n_shuffles=%d)…", args.n_shuffles)
    reference_bn = build_reference_bn(
        feature_df,
        trained_bn,
        n_shuffles=args.n_shuffles,
        pseudocount=args.pseudocount,
        seed=args.seed,
    )
    reference_bn.save(args.out_reference)

    logger.info("Training complete.")
    logger.info("  Trained BN   → %s", args.out_trained)
    logger.info("  Reference BN → %s", args.out_reference)


if __name__ == "__main__":
    main()

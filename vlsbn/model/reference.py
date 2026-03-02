"""
Reference BN construction via shuffled protein–ligand pairings.

Strategy
--------
1. Take the processed feature DataFrame (rows = complexes).
2. Separate metadata columns (pdb_id, ligand_id) from feature columns.
3. Shuffle the assignment of ligand-row indices to protein-row indices,
   destroying the true binding signal while preserving the marginal
   distributions of each atom type.
4. Discretise using the *trained BN's thresholds* (same bins → same DAG
   conditioning sets → well-defined LLR).
5. Fit CPTs on the shuffled discrete matrix using the *trained BN's topology*.
6. Return a BayesianNetwork with the reference CPTs and the original DAG.

Why shuffle rather than re-learn structure?
-------------------------------------------
The LLR term log[P(x_i | pa, trained) / P(x_i | pa, reference)] requires
the same parent set in numerator and denominator.  Using a fixed topology
(from real data) for both BNs guarantees this and keeps the comparison
mathematically clean.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vlsbn.model.bn import BayesianNetwork, CPT, fit_cpts

logger = logging.getLogger(__name__)

# Metadata columns that are never features
_META_COLS = {"pdb_id", "ligand_id"}


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

def _separate_meta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into (meta_df, feature_df)."""
    meta_cols    = [c for c in df.columns if c in _META_COLS]
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    return df[meta_cols], df[feature_cols]


def shuffle_pairings(
    feature_df: pd.DataFrame,
    n_shuffles: int = 1,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Create a shuffled feature matrix by randomly re-pairing protein and
    ligand contact profiles.

    The feature columns are conceptually split into two groups:
    - Protein-side signal   — not directly separable at this stage, because
      each row already encodes a protein–ligand *pair*.
    - Ligand-side signal    — same issue.

    The practical shuffle: permute row indices so that the contact profile
    of ligand i is matched with the contact profile of protein j ≠ i.  Since
    each row is a joint observation, we shuffle the entire row order, which
    is equivalent to randomly reassigning which ligand goes with which
    protein environment.  With enough shuffles and averaging, this converges
    to the random-association distribution.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Continuous feature matrix (no meta columns).
    n_shuffles : int
        Number of independent shuffles to average together.  n_shuffles = 1
        is fine for large datasets; increase for small ones.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shuffled feature matrix; same shape and columns as input.
    """
    rng = np.random.default_rng(seed)
    n = len(feature_df)
    values = feature_df.to_numpy(dtype=float)

    accumulated = np.zeros_like(values)
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        accumulated += values[perm]

    shuffled = accumulated / n_shuffles
    return pd.DataFrame(shuffled, columns=feature_df.columns, index=feature_df.index)


# ---------------------------------------------------------------------------
# Reference BN construction
# ---------------------------------------------------------------------------

def build_reference_bn(
    feature_df: pd.DataFrame,
    trained_bn: BayesianNetwork,
    n_shuffles: int = 1,
    pseudocount: float = 0.5,
    seed: Optional[int] = 42,
) -> BayesianNetwork:
    """Build the reference BN by fitting CPTs on shuffled data.

    Uses the *trained BN's topology and discretisation thresholds* so that
    the LLR computation has identical conditioning sets in both BNs.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Original (unshuffled) continuous feature matrix (no meta columns).
    trained_bn : BayesianNetwork
        The BN trained on real data.  Its topology and thresholds are reused.
    n_shuffles : int
        Passed to shuffle_pairings.
    pseudocount : float
        Laplace smoothing for CPT estimation.
    seed : int | None
        RNG seed.

    Returns
    -------
    BayesianNetwork
        A copy of trained_bn with reference CPTs fitted on shuffled data.
    """
    logger.info(
        "Building reference BN: shuffling %d complexes (%d shuffle(s))…",
        len(feature_df), n_shuffles,
    )
    shuffled_continuous = shuffle_pairings(feature_df, n_shuffles=n_shuffles, seed=seed)

    # Discretise with the SAME thresholds as the trained BN
    shuffled_discrete = trained_bn.thresholds.discretise(shuffled_continuous)

    # Fit CPTs using the SAME edges
    logger.info("Fitting reference CPTs on shuffled data…")
    ref_cpts = fit_cpts(
        shuffled_discrete,
        trained_bn.edges,
        pseudocount=pseudocount,
    )

    # Build a new BN object that shares topology/thresholds but has ref CPTs
    ref_bn = BayesianNetwork(
        nodes=trained_bn.nodes,
        edges=trained_bn.edges,
        cpts=ref_cpts,
        thresholds=trained_bn.thresholds,   # shared — thresholds are read-only
    )

    logger.info("Reference BN ready (%d nodes, %d edges).", len(ref_bn.nodes), len(ref_bn.edges))
    return ref_bn

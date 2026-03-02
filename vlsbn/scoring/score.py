"""
MI-weighted Log-Likelihood Ratio (LLR) scoring and coverage computation.

Scoring formula
---------------
    score = Σᵢ w(i) · log[ P(xᵢ | pa(xᵢ), BN_trained)
                          / P(xᵢ | pa(xᵢ), BN_reference) ]

where
    w(i) = I_trained(Xᵢ ; Pa(Xᵢ)) − I_reference(Xᵢ ; Pa(Xᵢ))

    I(X ; Pa(X)) is the conditional mutual information of node X and its
    parents under a given BN's CPT.

Positive score → interaction pattern more consistent with true binding
                 than random association.
Negative score → interaction pattern resembles random co-occurrence.

Coverage metric (reported separately)
--------------------------------------
    coverage = |{nodes with observed state > 0}  ∩  BN nodes|
               ─────────────────────────────────────────────────
               |{nodes with observed state > 0}|

Values < 1.0 indicate novel atom-type contacts not seen in training data
and signal reduced reliability of the primary score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vlsbn.model.bn import BayesianNetwork
from vlsbn.constants import N_STATES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mutual Information helpers
# ---------------------------------------------------------------------------

def _cpt_mi(cpt, bn: BayesianNetwork) -> float:
    """Compute I(X ; Pa(X)) in nats from a CPT.

    For a root node (no parents) MI = H(X) — the marginal entropy.
    For non-root nodes MI = H(X) − H(X | Pa(X)).

    The CPT table stores P(X | Pa(X)) row-by-row.  We need the *joint*
    P(X, Pa(X)) = P(X | Pa(X)) · P(Pa(X)) to compute MI, but P(Pa(X))
    is not stored explicitly.  We approximate by assuming uniform parent
    state distribution (equal weight per parent combo row), which is
    conservative but well-defined without a full forward pass.

    TODO: replace with exact MI via forward sampling once BaNDyT is
          integrated (it already computes joint distributions).

    Returns MI in nats (≥ 0).
    """
    table = cpt.table   # shape (n_parent_combos, N_STATES)

    n_rows = table.shape[0]
    # Approximate joint: uniform weight over parent combos
    joint = table / n_rows   # P(X, Pa=k) ≈ P(X|Pa=k) / n_parent_combos

    # Marginal P(X)
    p_x = joint.sum(axis=0)   # shape (N_STATES,)

    # MI = Σ_{x, pa} P(x, pa) · log[ P(x, pa) / (P(x) · P(pa)) ]
    # Since P(pa) = 1/n_rows under uniform assumption:
    mi = 0.0
    for row in range(n_rows):
        p_pa = 1.0 / n_rows
        for state in range(N_STATES):
            p_joint = joint[row, state]
            if p_joint <= 0:
                continue
            mi += p_joint * np.log(p_joint / (p_x[state] * p_pa + 1e-300))

    return max(float(mi), 0.0)


def compute_mi_weights(
    trained_bn: BayesianNetwork,
    reference_bn: BayesianNetwork,
) -> dict[str, float]:
    """Compute per-node MI-difference weights.

    w(i) = I_trained(Xᵢ ; Pa(Xᵢ)) − I_reference(Xᵢ ; Pa(Xᵢ))

    Nodes where real-data MI >> random-data MI receive high weight,
    meaning their CPT differences dominate the LLR.

    Parameters
    ----------
    trained_bn : BayesianNetwork
    reference_bn : BayesianNetwork

    Returns
    -------
    dict[str, float]
        Node name → weight (may be negative; clamp to 0 during scoring if desired).
    """
    weights: dict[str, float] = {}
    for node in trained_bn.nodes:
        mi_trained = _cpt_mi(trained_bn.cpts[node], trained_bn)
        mi_ref     = _cpt_mi(reference_bn.cpts[node], reference_bn)
        weights[node] = mi_trained - mi_ref
    return weights


# ---------------------------------------------------------------------------
# Scoring result container
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Output of score_complex().

    Attributes
    ----------
    llr : float
        MI-weighted log-likelihood ratio (primary score).
    coverage : float
        Fraction of observed contacts covered by the BN [0, 1].
    n_active_nodes : int
        Number of feature nodes with observed state > 0.
    node_contributions : dict[str, float]
        Per-node LLR contribution (before MI weighting) for interpretability.
    """
    llr:               float
    coverage:          float
    n_active_nodes:    int
    node_contributions: dict[str, float]

    def __repr__(self) -> str:
        return (
            f"ScoringResult(llr={self.llr:.4f}, coverage={self.coverage:.3f}, "
            f"active_nodes={self.n_active_nodes})"
        )


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_complex(
    feature_vector: dict[str, float],
    trained_bn: BayesianNetwork,
    reference_bn: BayesianNetwork,
    mi_weights: dict[str, float] | None = None,
    clamp_negative_weights: bool = False,
) -> ScoringResult:
    """Score a single protein–ligand complex.

    Parameters
    ----------
    feature_vector : dict[str, float]
        Continuous contact-density features from contacts.compute_features().
    trained_bn : BayesianNetwork
        BN trained on real crystal structures.
    reference_bn : BayesianNetwork
        Reference BN fitted on shuffled pairings (same topology).
    mi_weights : dict[str, float] | None
        Pre-computed MI-difference weights.  Computed on the fly if None.
    clamp_negative_weights : bool
        If True, nodes with w(i) < 0 are excluded from the LLR sum.

    Returns
    -------
    ScoringResult
    """
    if mi_weights is None:
        mi_weights = compute_mi_weights(trained_bn, reference_bn)

    # Discretise the input feature vector using trained BN thresholds
    feat_series = pd.Series(feature_vector)
    feat_df     = feat_series.to_frame().T
    discrete_df = trained_bn.thresholds.discretise(feat_df)

    bn_node_set = set(trained_bn.nodes)
    active_nodes   = {k for k, v in feature_vector.items() if v > 0}
    covered_nodes  = active_nodes & bn_node_set

    coverage = (
        len(covered_nodes) / len(active_nodes) if active_nodes else 1.0
    )

    llr_total = 0.0
    node_contributions: dict[str, float] = {}

    for node in trained_bn.nodes:
        if node not in discrete_df.columns:
            continue

        x_state = int(discrete_df[node].iloc[0])
        parents = trained_bn.parents_of(node)
        pa_states = tuple(
            int(discrete_df[p].iloc[0]) if p in discrete_df.columns else 0
            for p in parents
        )

        log_p_trained   = np.log(trained_bn.cpts[node].prob(x_state, pa_states))
        log_p_reference = np.log(reference_bn.cpts[node].prob(x_state, pa_states))
        node_llr = log_p_trained - log_p_reference

        w = mi_weights.get(node, 0.0)
        if clamp_negative_weights:
            w = max(w, 0.0)

        node_contributions[node] = node_llr
        llr_total += w * node_llr

    return ScoringResult(
        llr=llr_total,
        coverage=coverage,
        n_active_nodes=len(active_nodes),
        node_contributions=node_contributions,
    )


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_dataframe(
    feature_df: pd.DataFrame,
    trained_bn: BayesianNetwork,
    reference_bn: BayesianNetwork,
    clamp_negative_weights: bool = False,
) -> pd.DataFrame:
    """Score all complexes in a feature DataFrame.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Rows = complexes; may contain pdb_id / ligand_id meta columns.
    trained_bn : BayesianNetwork
    reference_bn : BayesianNetwork

    Returns
    -------
    pd.DataFrame
        Columns: pdb_id, ligand_id (if present), llr, coverage, n_active_nodes.
    """
    meta_cols    = [c for c in ("pdb_id", "ligand_id") if c in feature_df.columns]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    mi_weights = compute_mi_weights(trained_bn, reference_bn)

    records = []
    for _, row in feature_df.iterrows():
        feat_vec = row[feature_cols].to_dict()
        result   = score_complex(
            feat_vec, trained_bn, reference_bn,
            mi_weights=mi_weights,
            clamp_negative_weights=clamp_negative_weights,
        )
        rec = {c: row[c] for c in meta_cols}
        rec["llr"]            = result.llr
        rec["coverage"]       = result.coverage
        rec["n_active_nodes"] = result.n_active_nodes
        records.append(rec)

    return pd.DataFrame(records)

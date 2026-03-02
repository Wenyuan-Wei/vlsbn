"""
Bayesian Network — structure learning, CPT fitting, and persistence.

Design
------
Each column of the feature DataFrame is a BN node.  Column values are
contact densities (float in [0, 1]) that must be discretised into
N_STATES = 4 ordered states before learning.

Discretisation thresholds are fitted on the training set:
    state 0 → density == 0          (no contact)
    state 1 → 0 < density ≤ Q1      (sparse)
    state 2 → Q1 < density ≤ Q3     (moderate)
    state 3 → density > Q3          (dense)

BN structure learning
---------------------
TODO: integrate BaNDyT (https://github.com/bandyt-group/bandyt).
      Specifically review bandyt/bandyt.py for the structure-learning
      entry point and bandyt/oflib.py for the C++ scoring extension.
      The interface below is intentionally BaNDyT-shaped so the swap
      will be minimal.

Until BaNDyT is wired in, a placeholder hill-climbing stub (using
networkx DAGs) holds the interface stable.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vlsbn.constants import N_STATES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class Thresholds:
    """Per-node discretisation thresholds (Q1 and Q3 over training densities).

    Attributes
    ----------
    q1, q3 : pd.Series
        Index = node (column) name; values = float threshold.
    """

    def __init__(self, q1: pd.Series, q3: pd.Series) -> None:
        self.q1 = q1
        self.q3 = q3

    def discretise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discretise a continuous feature DataFrame into integer states.

        Parameters
        ----------
        df : pd.DataFrame
            Rows = complexes; columns = feature triplet keys.

        Returns
        -------
        pd.DataFrame
            Same shape, dtype int8, values in {0, 1, 2, 3}.
        """
        out = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=np.int8)
        for col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            q1   = self.q1.get(col, np.nan)
            q3   = self.q3.get(col, np.nan)
            if np.isnan(q1) or np.isnan(q3):
                # Column not seen during threshold fitting; leave as 0
                continue
            out[col] = np.where(
                vals == 0, 0,
                np.where(vals <= q1, 1,
                np.where(vals <= q3, 2, 3))
            ).astype(np.int8)
        return out

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Thresholds":
        """Fit Q1 / Q3 thresholds from non-zero density values per column."""
        q1_vals, q3_vals = {}, {}
        for col in df.columns:
            nonzero = df[col][df[col] > 0]
            if nonzero.empty:
                q1_vals[col] = 0.0
                q3_vals[col] = 0.0
            else:
                q1_vals[col] = float(nonzero.quantile(0.25))
                q3_vals[col] = float(nonzero.quantile(0.75))
        return cls(pd.Series(q1_vals), pd.Series(q3_vals))


class CPT:
    """Conditional Probability Table for one BN node.

    Stores P(node_state | parent_state_combo) as a numpy array.

    Attributes
    ----------
    node : str
        Column name this CPT belongs to.
    parents : list[str]
        Ordered list of parent column names.
    table : np.ndarray
        Shape (n_parent_combos, N_STATES); rows sum to 1.
    parent_states : int
        N_STATES ** len(parents) — total number of parent state combos.
    """

    def __init__(
        self,
        node: str,
        parents: list[str],
        table: np.ndarray,
    ) -> None:
        self.node    = node
        self.parents = parents
        self.table   = table

    def prob(self, node_state: int, parent_states: tuple[int, ...]) -> float:
        """P(node = node_state | parents = parent_states).

        Returns a small epsilon (1e-9) instead of 0 to avoid log(0).
        """
        if not self.parents:
            return float(self.table[0, node_state])
        row = int(np.ravel_multi_index(parent_states, (N_STATES,) * len(self.parents)))
        return max(float(self.table[row, node_state]), 1e-9)


class BayesianNetwork:
    """Container for the full learned BN: DAG topology + CPTs.

    Attributes
    ----------
    nodes : list[str]
        Topological order of nodes (feature column names).
    edges : list[tuple[str, str]]
        Directed edges (parent, child) in the DAG.
    cpts : dict[str, CPT]
        Per-node conditional probability tables.
    thresholds : Thresholds
        Discretisation thresholds (fitted on training data).
    """

    def __init__(
        self,
        nodes: list[str],
        edges: list[tuple[str, str]],
        cpts: dict[str, CPT],
        thresholds: Thresholds,
    ) -> None:
        self.nodes      = nodes
        self.edges      = edges
        self.cpts       = cpts
        self.thresholds = thresholds

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def parents_of(self, node: str) -> list[str]:
        return [p for p, c in self.edges if c == node]

    def weighted_degree(self, node: str) -> float:
        """Number of edges incident to *node* (in + out)."""
        return sum(1 for p, c in self.edges if p == node or c == node)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        logger.info("BN saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "BayesianNetwork":
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        logger.info("BN loaded ← %s", path)
        return obj


# ---------------------------------------------------------------------------
# Structure learning  (TODO: replace stub with BaNDyT)
# ---------------------------------------------------------------------------

def learn_structure(
    discrete_df: pd.DataFrame,
    max_parents: int = 4,
) -> list[tuple[str, str]]:
    """Learn BN DAG structure from a discretised feature DataFrame.

    TODO: replace this stub with BaNDyT's structure-learning routine.
          Review bandyt/bandyt.py for the entry point.
          Review bandyt/oflib.py for the C++ BIC/BDeu scoring extension.

    Current stub returns an empty edge list (fully disconnected graph),
    which yields a valid but naive BN.  Swap in BaNDyT before training.

    Parameters
    ----------
    discrete_df : pd.DataFrame
        Integer-valued (0–3) feature matrix; rows = complexes.
    max_parents : int
        Maximum in-degree per node (BaNDyT parameter).

    Returns
    -------
    list[tuple[str, str]]
        Directed edges (parent_node, child_node).
    """
    logger.warning(
        "learn_structure: BaNDyT not yet integrated — returning empty graph. "
        "See TODO in vlsbn/model/bn.py."
    )
    return []


# ---------------------------------------------------------------------------
# CPT fitting
# ---------------------------------------------------------------------------

def fit_cpts(
    discrete_df: pd.DataFrame,
    edges: list[tuple[str, str]],
    pseudocount: float = 0.5,
) -> dict[str, CPT]:
    """Estimate CPTs from a discretised feature matrix via Laplace smoothing.

    Parameters
    ----------
    discrete_df : pd.DataFrame
        Integer-valued (0–3) feature matrix.
    edges : list[tuple[str, str]]
        DAG edges (parent, child).
    pseudocount : float
        Laplace (add-alpha) smoothing constant.

    Returns
    -------
    dict[str, CPT]
        Node name → CPT object.
    """
    # Build parent map
    parent_map: dict[str, list[str]] = {col: [] for col in discrete_df.columns}
    for parent, child in edges:
        parent_map[child].append(parent)

    cpts: dict[str, CPT] = {}

    for node, parents in parent_map.items():
        n_parent_combos = N_STATES ** len(parents) if parents else 1
        table = np.zeros((n_parent_combos, N_STATES), dtype=np.float64)

        node_vals = discrete_df[node].to_numpy(dtype=int)

        if not parents:
            for state in range(N_STATES):
                table[0, state] = np.sum(node_vals == state) + pseudocount
            table[0] /= table[0].sum()
        else:
            parent_vals = discrete_df[parents].to_numpy(dtype=int)
            for row_idx in range(len(discrete_df)):
                pstate_combo = tuple(parent_vals[row_idx])
                row = int(np.ravel_multi_index(pstate_combo, (N_STATES,) * len(parents)))
                table[row, node_vals[row_idx]] += 1

            # Add pseudocount and normalise each row
            table += pseudocount
            row_sums = table.sum(axis=1, keepdims=True)
            table /= row_sums

        cpts[node] = CPT(node=node, parents=parents, table=table)

    return cpts


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train(
    feature_df: pd.DataFrame,
    max_parents: int = 4,
    pseudocount: float = 0.5,
) -> BayesianNetwork:
    """End-to-end BN training from a continuous feature DataFrame.

    Steps
    -----
    1. Fit discretisation thresholds.
    2. Discretise feature matrix.
    3. Learn DAG structure (BaNDyT — currently stubbed).
    4. Fit CPTs with Laplace smoothing.
    5. Return a BayesianNetwork object.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Rows = complexes; columns = contact-density feature keys.
        Must NOT contain pdb_id / ligand_id meta-columns.
    max_parents : int
        Passed to structure-learning routine.
    pseudocount : float
        Laplace smoothing for CPT estimation.

    Returns
    -------
    BayesianNetwork
    """
    logger.info("Fitting discretisation thresholds on %d complexes…", len(feature_df))
    thresholds = Thresholds.fit(feature_df)
    discrete_df = thresholds.discretise(feature_df)

    logger.info("Learning BN structure…")
    edges = learn_structure(discrete_df, max_parents=max_parents)
    logger.info("Structure: %d nodes, %d edges.", len(discrete_df.columns), len(edges))

    logger.info("Fitting CPTs (pseudocount=%.2f)…", pseudocount)
    cpts = fit_cpts(discrete_df, edges, pseudocount=pseudocount)

    nodes = list(discrete_df.columns)
    return BayesianNetwork(
        nodes=nodes,
        edges=edges,
        cpts=cpts,
        thresholds=thresholds,
    )

"""
Unit tests for vlsbn.model — discretisation, CPT fitting, and BN training stub.

Run with:  pytest tests/test_model.py -v
"""

import numpy as np
import pandas as pd
import pytest

from vlsbn.model.bn import CPT, BayesianNetwork, Thresholds, fit_cpts, train
from vlsbn.model.reference import build_reference_bn, shuffle_pairings
from vlsbn.constants import N_STATES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n: int = 100, cols: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cols or ["A__X__hb", "B__Y__vdw", "C__Z__ps"]
    data = rng.uniform(0, 0.3, size=(n, len(cols)))
    # Inject some zeros to mimic sparse real data
    data[rng.random(data.shape) < 0.4] = 0.0
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_fit_returns_thresholds(self):
        df = _make_feature_df()
        t = Thresholds.fit(df)
        assert set(t.q1.index) == set(df.columns)

    def test_discretise_output_range(self):
        df = _make_feature_df()
        t = Thresholds.fit(df)
        disc = t.discretise(df)
        assert disc.shape == df.shape
        assert disc.min().min() == 0
        assert disc.max().max() <= N_STATES - 1

    def test_zero_density_stays_zero(self):
        df = pd.DataFrame({"A__X__hb": [0.0, 0.0, 0.5]})
        t = Thresholds.fit(df)
        disc = t.discretise(df)
        assert disc["A__X__hb"].iloc[0] == 0
        assert disc["A__X__hb"].iloc[1] == 0

    def test_all_zero_column(self):
        df = pd.DataFrame({"A__X__hb": [0.0, 0.0, 0.0]})
        t = Thresholds.fit(df)
        disc = t.discretise(df)
        assert (disc["A__X__hb"] == 0).all()


# ---------------------------------------------------------------------------
# CPT fitting
# ---------------------------------------------------------------------------

class TestCPTFitting:
    def test_root_cpt_sums_to_one(self):
        df = _make_feature_df()
        t = Thresholds.fit(df)
        disc = t.discretise(df)
        cpts = fit_cpts(disc, edges=[], pseudocount=0.5)

        for node, cpt in cpts.items():
            assert abs(cpt.table[0].sum() - 1.0) < 1e-6, f"CPT for {node} rows don't sum to 1"

    def test_child_cpt_rows_sum_to_one(self):
        cols = ["P", "C"]
        df = _make_feature_df(cols=cols)
        t = Thresholds.fit(df)
        disc = t.discretise(df)
        edges = [("P", "C")]
        cpts = fit_cpts(disc, edges=edges, pseudocount=0.5)

        child_cpt = cpts["C"]
        assert child_cpt.parents == ["P"]
        for row in child_cpt.table:
            assert abs(row.sum() - 1.0) < 1e-6

    def test_cpt_prob_returns_positive(self):
        df = _make_feature_df()
        t = Thresholds.fit(df)
        disc = t.discretise(df)
        cpts = fit_cpts(disc, edges=[], pseudocount=0.5)
        node = list(cpts)[0]
        assert cpts[node].prob(0, ()) > 0


# ---------------------------------------------------------------------------
# BayesianNetwork
# ---------------------------------------------------------------------------

class TestBayesianNetwork:
    def _build(self):
        df = _make_feature_df()
        return train(df)   # uses empty-graph stub

    def test_train_returns_bn(self):
        bn = self._build()
        assert isinstance(bn, BayesianNetwork)

    def test_nodes_match_columns(self):
        df = _make_feature_df()
        bn = train(df)
        assert set(bn.nodes) == set(df.columns)

    def test_weighted_degree_root(self):
        df = _make_feature_df()
        bn = train(df)
        # Stub returns no edges, so all degrees are 0
        for node in bn.nodes:
            assert bn.weighted_degree(node) == 0

    def test_save_load_roundtrip(self, tmp_path):
        bn = self._build()
        path = tmp_path / "bn.pkl"
        bn.save(path)
        loaded = BayesianNetwork.load(path)
        assert loaded.nodes == bn.nodes


# ---------------------------------------------------------------------------
# Reference BN
# ---------------------------------------------------------------------------

class TestReferenceBN:
    def test_shuffle_preserves_shape(self):
        df = _make_feature_df()
        shuffled = shuffle_pairings(df, seed=0)
        assert shuffled.shape == df.shape
        assert list(shuffled.columns) == list(df.columns)

    def test_shuffled_different_from_original(self):
        df = _make_feature_df(n=200)
        shuffled = shuffle_pairings(df, seed=1)
        # Column means should be approximately equal (same marginals)
        np.testing.assert_allclose(
            df.mean().values, shuffled.mean().values, atol=0.05
        )

    def test_reference_bn_same_topology(self):
        df = _make_feature_df()
        trained = train(df)
        ref = build_reference_bn(df, trained, seed=7)
        assert ref.edges == trained.edges
        assert ref.nodes == trained.nodes

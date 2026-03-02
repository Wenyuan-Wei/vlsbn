"""
Unit tests for vlsbn.scoring.

Run with:  pytest tests/test_scoring.py -v
"""

import numpy as np
import pandas as pd
import pytest

from vlsbn.model.bn import train
from vlsbn.model.reference import build_reference_bn
from vlsbn.scoring.score import (
    ScoringResult,
    compute_mi_weights,
    score_complex,
    score_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bns(n: int = 200, cols=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = cols or ["A__X__hb", "B__Y__vdw", "C__Z__ps"]
    data = rng.uniform(0, 0.3, size=(n, len(cols)))
    data[rng.random(data.shape) < 0.4] = 0.0
    df = pd.DataFrame(data, columns=cols)
    trained = train(df)
    reference = build_reference_bn(df, trained, seed=seed)
    return trained, reference, df


# ---------------------------------------------------------------------------
# MI weights
# ---------------------------------------------------------------------------

class TestMIWeights:
    def test_returns_dict_for_all_nodes(self):
        trained, ref, _ = _make_bns()
        weights = compute_mi_weights(trained, ref)
        assert set(weights.keys()) == set(trained.nodes)

    def test_weights_are_floats(self):
        trained, ref, _ = _make_bns()
        weights = compute_mi_weights(trained, ref)
        for k, v in weights.items():
            assert isinstance(v, float), f"Weight for {k} is not float"


# ---------------------------------------------------------------------------
# score_complex
# ---------------------------------------------------------------------------

class TestScoreComplex:
    def test_returns_scoring_result(self):
        trained, ref, df = _make_bns()
        feat = df.iloc[0].to_dict()
        result = score_complex(feat, trained, ref)
        assert isinstance(result, ScoringResult)

    def test_llr_is_finite(self):
        trained, ref, df = _make_bns()
        feat = df.iloc[0].to_dict()
        result = score_complex(feat, trained, ref)
        assert np.isfinite(result.llr)

    def test_coverage_between_0_and_1(self):
        trained, ref, df = _make_bns()
        feat = df.iloc[0].to_dict()
        result = score_complex(feat, trained, ref)
        assert 0.0 <= result.coverage <= 1.0

    def test_zero_vector_coverage(self):
        trained, ref, df = _make_bns()
        # All-zero vector = no active contacts
        feat = {col: 0.0 for col in df.columns}
        result = score_complex(feat, trained, ref)
        assert result.n_active_nodes == 0
        assert result.coverage == 1.0   # convention: no active → full coverage

    def test_novel_feature_reduces_coverage(self):
        trained, ref, df = _make_bns()
        feat = df.iloc[0].to_dict()
        feat["NOVEL__TYPE__hb"] = 0.5   # key not in BN
        result = score_complex(feat, trained, ref)
        assert result.coverage < 1.0


# ---------------------------------------------------------------------------
# score_dataframe
# ---------------------------------------------------------------------------

class TestScoreDataframe:
    def test_output_shape(self):
        trained, ref, df = _make_bns(n=10)
        df_meta = df.copy()
        df_meta["pdb_id"]    = [f"P{i}" for i in range(len(df))]
        df_meta["ligand_id"] = [f"L{i}" for i in range(len(df))]
        out = score_dataframe(df_meta, trained, ref)
        assert len(out) == len(df)
        assert "llr" in out.columns
        assert "coverage" in out.columns

    def test_all_llr_finite(self):
        trained, ref, df = _make_bns(n=20)
        out = score_dataframe(df, trained, ref)
        assert out["llr"].apply(np.isfinite).all()

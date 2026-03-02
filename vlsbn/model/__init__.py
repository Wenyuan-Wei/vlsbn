"""
Model subpackage — Bayesian Network construction and reference model.

bn.py         Learn BN topology from real-data feature matrix; fit CPTs.
reference.py  Generate shuffled pairings; fit reference CPTs on same topology.

Note: BN structure learning is delegated to BaNDyT
(https://github.com/bandyt-group/bandyt). The interfaces here are stable;
the BaNDyT integration is marked TODO and will be completed once the repo
has been reviewed in detail.
"""

# VLS_BN — Project Notes

## Overview

A knowledge-based protein-ligand scoring function built on a **Bayesian Network (BN)** framework, inspired by BaNDyT. The core idea is to reduce protein-ligand interactions to the level of **atom-type pairs**, derive their statistical properties from all available crystal structures in the RCSB PDB, and encode conditional dependencies between interaction types via a BN.

The eventual goal is a scoring function comparable in purpose to those used in conventional docking software (e.g., Glide).

---

## Conceptual Foundation

### Why Atom-Type Pairs?
Proteins and small molecules use a finite set of chemical elements and hybridization states, yielding a bounded number of distinct atom types. All protein-ligand interactions can therefore be reduced to a finite vocabulary of atom-type pair contacts. This makes the problem tractable and data-driven.

### Why a Bayesian Network?
Conventional knowledge-based scoring functions (PMF, DrugScore, IT-Score, DSX) treat atom-pair interactions as **independent**. A BN can capture **conditional dependencies** between interaction types — e.g., the probability of a hydrophobic contact may depend on whether a hydrogen bond is already present nearby. This is the key methodological novelty over classical statistical potentials.

### Adapting BaNDyT
BaNDyT uses BNs to analyze molecular dynamics trajectories, where each frame is a sample. Here, the analogy is:
- MD frames → crystal structures from PDB (each complex = one sample)
- Conformational variables → atom-pair interaction descriptors

---

## Training Data

- **Source**: RCSB PDB (bulk download or `rcsb_search` API)
- **Filters** (proposed):
  - Resolution ≤ 2.5 Å
  - R-free ≤ 0.35
  - Exclude DNA/RNA-bound entries
  - Exclude covalent ligands (optional, revisit later)
  - Exclude crystallographic artifacts (PEG, glycerol, common buffer molecules)
- **Scale**: Tens of thousands of protein-ligand complexes expected after filtering

---

## Plan of Action

### Phase 1 — Data Pipeline
- Download all PDB entries containing protein-ligand complexes
- Apply quality filters listed above
- Parse structures: extract protein heavy-atom coordinates + types, ligand heavy-atom coordinates + types

### Phase 2 — Atom Typing & Feature Extraction
- Define atom type taxonomy for proteins (SYBYL-like or custom minimal set)
- Assign ligand atom types using RDKit from embedded SMILES/SDF
- Run getcontacts to compute labeled contacts per complex
- Compute contact density per (protein atom type, ligand atom type, interaction type) triplet
- Discretize densities into BN node states using training-set quantiles

### Phase 3 — Bayesian Network Construction
- One node per (protein atom type, ligand atom type, interaction type) triplet
- Structure learning: score-based (BIC/BDeu with greedy hill-climbing) or constraint-based (PC algorithm)
- Parameter learning: conditional probability tables (discrete nodes)

### Phase 4 — Scoring & Validation
- Define scalar score from BN output (log-likelihood ratio, marginal probability, etc.) — **TBD**
- Validate against experimental binding affinities from PDBbind

---

## Node Design — Finalized

### Node Identity
Each BN node represents a unique **(protein atom type, ligand atom type, interaction type)** triplet.

Example nodes:
- `(N_amide, O_carbonyl, hb)` — hydrogen bond between backbone N and ligand carbonyl
- `(C_aromatic, C_aromatic, ps)` — pi-stacking between protein and ligand aromatics
- `(N_charged, O_charged, sb)` — salt bridge

Interaction types sourced from getcontacts: `hb`, `sb`, `pc`, `ps`, `ts`, `vdw`

### Node Value — Contact Density
Raw contact counts are normalized to remove the confound of ligand/protein size:

```
density(X, Y, itype) = count of observed contacts(X, Y, itype)
                       ────────────────────────────────────────
                       count of (X, Y) atom pairs within shell radius (10 Å)
```

This gives a dimensionless value in [0, 1]: the fraction of available X–Y pairs that are actually in contact of type `itype`. Physical interpretation: *"given these atom types are in proximity, how often do they form this interaction?"*

### Node States — Discretization
Contact density is discretized into 4 states for BN compatibility:

| State | Condition | Meaning |
|-------|-----------|---------|
| `0` | density = 0 | No contact observed |
| `1` | 0 < density ≤ Q1 | Sparse contact |
| `2` | Q1 < density ≤ Q3 | Moderate contact |
| `3` | density > Q3 | Dense contact |

Quantile thresholds (Q1, Q3) are computed across the full training set for each node, ensuring balanced state occupancy.

### getcontacts Cutoffs (Reference)
Cutoffs from `argparsers.py` used for contact detection:

| Interaction | Distance Cutoff | Angle Cutoff |
|-------------|----------------|--------------|
| Hydrogen bond (`hb`) | ≤ 3.5 Å | ≥ 70° (≥ 180° for structures) |
| Salt bridge (`sb`) | ≤ 4.0 Å | — |
| T-stacking (`ts`) | ≤ 5.0 Å | 30° / 45° |
| Pi–cation (`pc`) | ≤ 6.0 Å | ≥ 60° |
| Pi–stacking (`ps`) | ≤ 7.0 Å | 30° / 45° |
| Van der Waals (`vdw`) | VDW radii sum + 0.5 Å | — |

---

## Scoring Design — Finalized

### Primary Score: Importance-Weighted Log-Likelihood Ratio (LLR)

```
score = Σᵢ w(i) * log [ P(Xᵢ | parents(Xᵢ), BN_trained) / P(Xᵢ | parents(Xᵢ), BN_reference) ]
```

- **w(i)** = weighted degree of node i in the trained BN — highly connected nodes (those with strong conditional influence on others) contribute more to the final score
- **Positive score** → interaction pattern is more consistent with true binding than random association
- **Negative score** → interaction pattern resembles random co-occurrence

### Reference BN

The reference model is a **fully trained BN on shuffled protein-ligand pairings**: take the same PDB structures but randomly reassign which ligand goes with which protein, then train a BN on the shuffled dataset. This:
- Preserves marginal distributions of each atom type in isolation
- Destroys the true binding signal
- Captures realistic "random association" dependencies (not just a naive independent model)
- Answers: *"how does this interaction pattern compare to random co-location of these atom types?"*

### Secondary Metric: Coverage Score

Tracks fragmentation — how many of the new complex's contacts are actually represented in the trained BN:

```
coverage = count(complex contacts present in BN) / count(complex total contacts)
```

- Low coverage → unusual interaction chemistry not well-represented in training data → score is less reliable
- Reported alongside the primary score as a confidence indicator, not folded into it

### Future Experiment: Supervised Scoring Layer

Once the BN is built, we can experiment with treating the LLR and coverage as features and fitting a supervised model against experimental binding affinities (PDBbind). This requires labeled data and is deferred until the generative model is validated.

### Reference BN — Resolved Design

- **Topology**: learned from **real data only** (captures true binding dependencies)
- **BN_trained CPTs**: fitted on real data using that topology
- **BN_reference CPTs**: fitted on shuffled data using the **same topology**
- **Node weights**: `w(i) = I_trained(Xᵢ; Pa(Xᵢ)) − I_reference(Xᵢ; Pa(Xᵢ))` — conditional mutual information difference; nodes more discriminative between binding and random get higher weight

### BaNDyT — Reusable Code

The BaNDyT repo (https://github.com/bandyt-group/bandyt) is available for direct reuse (project co-author). Key components to mine when implementing:
- **Network generation**: `bandyt/bandyt.py` — main BN structure learning
- **Scoring/rescoring**: review for any LLR or probability evaluation utilities
- **C++ extension**: `bandyt/ofext.cpp` + `oflib.py` — performance-critical computations; reuse directly
- **Dependencies already established**: pandas, numpy, igraph, networkx, matplotlib, seaborn

Revisit this repo in detail when implementing Phase 3 (BN construction) and Phase 4 (scoring).

---

## Codebase Structure

```
Project_VLS_BN/
├── pyproject.toml
├── project_notes.md
├── data/
│   ├── raw/              downloaded PDB files
│   ├── filtered/         post-quality-filter (reserved)
│   └── processed/        contact_features.parquet
├── vlsbn/
│   ├── constants.py      all magic numbers, paths, cutoffs
│   ├── pipeline/
│   │   ├── fetch.py      RCSB search API + bulk download
│   │   ├── parse.py      BioPython PDB parsing → AtomRecord / ParsedComplex
│   │   ├── atomtypes.py  protein lookup table + RDKit ligand typer
│   │   └── contacts.py   getcontacts wrapper + density computation
│   ├── model/
│   │   ├── bn.py         Thresholds, CPT, BayesianNetwork, train()
│   │   │                 *** learn_structure() is a stub — needs BaNDyT ***
│   │   └── reference.py  shuffle_pairings() + build_reference_bn()
│   └── scoring/
│       └── score.py      compute_mi_weights(), score_complex(), score_dataframe()
├── scripts/
│   ├── run_pipeline.py   CLI: download → parse → features → parquet
│   └── train_bn.py       CLI: features → trained BN + reference BN
└── tests/
    ├── test_pipeline.py  atom typing, ParsedComplex
    ├── test_model.py     Thresholds, CPT, BN train, reference BN
    └── test_scoring.py   MI weights, score_complex, score_dataframe
```

## Key Challenges

1. **BN structure learning at scale** — NP-hard in general; approximations required.
2. **Sparse data for rare atom-type triplets** — Some (protein type, ligand type, itype) combinations will be rare in PDB; smoothing/regularization needed.
3. **Scoring interpretation** — The link between BN output and binding affinity is not yet defined.

---

## Status

- [x] Concept defined
- [x] Feasibility assessed
- [x] High-level plan drafted
- [x] Node representation finalized — (protein atom type, ligand atom type, interaction type) → discretized contact density
- [x] Scoring output design — importance-weighted LLR vs. shuffled-data reference BN; coverage as secondary metric
- [x] Codebase scaffolded — see directory structure below
- [ ] BaNDyT integration in vlsbn/model/bn.py (learn_structure stub)
- [ ] Data pipeline run on full PDB
- [ ] BN trained and validated against PDBbind

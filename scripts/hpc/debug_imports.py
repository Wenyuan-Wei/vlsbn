#!/usr/bin/env python3
"""
Import-isolation diagnostic for VLS_BN on HPC.

Tests every import used by hpc_process.py in a separate subprocess so that
a segfault in one library does not prevent the others from being tested.

Run on the HPC (after conda activate vlsbn) from the repo root:
    python scripts/hpc/debug_imports.py

Exit codes reported per test:
    0   = OK
    139 = Segmentation fault (SIGSEGV)
    1   = ImportError / Python exception
    other = some other failure
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)

# Each entry: (label, python_snippet)
# Snippets are executed in isolated subprocesses so a crash in one does not
# affect the others.
TESTS: list[tuple[str, str]] = [
    # --- stdlib (should always pass) ---
    ("stdlib: os / sys / pathlib",
     "import os, sys, pathlib; print('ok')"),

    # --- numeric stack ---
    ("numpy",
     "import numpy; print('ok', numpy.__version__)"),
    ("scipy",
     "import scipy; print('ok', scipy.__version__)"),
    ("pandas",
     "import pandas; print('ok', pandas.__version__)"),

    # --- biopython ---
    ("biopython (Bio)",
     "import Bio; print('ok', Bio.__version__)"),
    ("biopython MMCIFParser",
     "from Bio.PDB import MMCIFParser, PDBIO; print('ok')"),

    # --- rdkit (most likely culprit on HPC) ---
    ("rdkit import",
     "import rdkit; print('ok', rdkit.__version__)"),
    ("rdkit.Chem",
     "from rdkit import Chem; print('ok')"),
    ("rdkit.Chem.rdchem (used in atomtypes.py)",
     "from rdkit.Chem import rdchem; print('ok')"),

    # --- vmd-python (installed but not directly imported by our code) ---
    ("vmd-python",
     "import vmd; print('ok')"),

    # --- project modules (in import order) ---
    ("vlsbn.constants",
     f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
     "import vlsbn.constants; print('ok')"),
    ("vlsbn.pipeline.atomtypes",
     f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
     "from vlsbn.pipeline.atomtypes import PROTEIN_ATOM_TYPES; print('ok')"),
    ("vlsbn.pipeline.parse",
     f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
     "from vlsbn.pipeline.parse import parse_pdb; print('ok')"),
    ("vlsbn.pipeline.contacts",
     f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
     "from vlsbn.pipeline.contacts import compute_features; print('ok')"),
    ("vlsbn.pipeline.fetch",
     f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
     "from vlsbn.pipeline.fetch import download_pdb; print('ok')"),
    ("full hpc_process import block",
     f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
     "import pandas; "
     "from vlsbn.pipeline.contacts import compute_features; "
     "from vlsbn.pipeline.fetch import download_pdb; "
     "from vlsbn.pipeline.parse import parse_pdb; "
     "print('ok')"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def run_test(label: str, snippet: str) -> bool:
    """Run snippet in a subprocess; return True if it exited 0."""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
    )
    code = result.returncode
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if code == 0:
        status = f"{GREEN}PASS{RESET}"
        detail = stdout
    elif code == -11 or code == 139:
        status = f"{RED}SEGFAULT{RESET}"
        detail = f"exit {code}"
    elif code != 0:
        status = f"{RED}FAIL (exit {code}){RESET}"
        detail = (stderr or stdout)[:200]
    else:
        status = f"{GREEN}PASS{RESET}"
        detail = stdout

    print(f"  {status:<30}  {label}")
    if detail and code != 0:
        # Indent error details
        for line in detail.splitlines()[:5]:
            print(f"              {line}")
    sys.stdout.flush()
    return code == 0


def main() -> None:
    print()
    print("=" * 65)
    print("  VLS_BN import diagnostic")
    print(f"  Python: {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}")
    print("=" * 65)
    print()

    passed = failed = 0
    for label, snippet in TESTS:
        ok = run_test(label, snippet)
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 65)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 65)
    print()

    if failed:
        print("Suggested fix if rdkit segfaults:")
        print(textwrap.dedent("""
            conda remove rdkit
            conda install -c conda-forge rdkit

            Then re-run this script to confirm.
        """))
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Slurm submission scripts for the VLS_BN HPC pipeline.

Run this script locally (or on the login node) to produce a `slurm/`
directory containing three job scripts.  Stages are NOT chained
automatically — submit each one manually after the previous completes.

    Stage 0  00_fetch_ids.sh   — fetch PDB IDs from RCSB → data/pdb_ids.txt
    Stage 1  01_process.sh     — array job: one PDB per task, 5-step pipeline
    Stage 2  02_merge_train.sh — merge chunk parquets + train BN

Pipeline steps inside 01_process.sh
-------------------------------------
  Step 1+2  step1_prepare.py     download + parse + write temp PDBs + delete original
  Step 3    step2_protonate.py   add H atoms with RDKit / reduce
  Step 4    get_static_contacts.py  (explicit bash command; runs under getcontacts env)
  Step 5    step3_parse_contacts.py  parse TSV → raw contacts CSV

Set STOP_AFTER=N (1–5) to halt after any step for debugging:
  sbatch --export=ALL,STOP_AFTER=3 slurm/01_process.sh

Usage
-----
    python scripts/hpc/generate_slurm.py \\
        --partition gpu           \\
        --account   mygroup       \\
        --conda-env vlsbn         \\
        --getcontacts ~/getcontacts/get_static_contacts.py \\
        --getcontacts-conda-env vmd-python   # conda env that has vmd-python

Then on the HPC:
    cd ~/Project_VLS_BN
    sbatch slurm/00_fetch_ids.sh
    # After it finishes, check data/pdb_ids.txt, then:
    sbatch --array=0-<N_IDS-1>%200 slurm/01_process.sh
    # After array finishes:
    sbatch slurm/02_merge_train.sh
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Slurm scripts for VLS_BN pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--partition",        default="cpu",
                   help="Slurm partition name")
    p.add_argument("--account",          default=None,
                   help="Slurm account/project (omit if not required by your cluster)")
    p.add_argument("--conda-env",        default="vlsbn",
                   help="Conda environment name for the main pipeline (steps 1–3, 5)")
    p.add_argument("--walltime-fetch",   default="0:30:00",
                   help="Walltime for stage 0 (fetch IDs)")
    p.add_argument("--walltime-process", default="4:00:00",
                   help="Walltime per array task (stage 1)")
    p.add_argument("--walltime-merge",   default="2:00:00",
                   help="Walltime for stage 2 (merge + train)")
    p.add_argument("--mem-process",      default="4G",
                   help="Memory per array task")
    p.add_argument("--mem-merge",        default="32G",
                   help="Memory for merge+train job")
    p.add_argument("--cpus-process",     type=int, default=1,
                   help="CPUs per array task (getcontacts is single-threaded)")
    p.add_argument("--getcontacts",
                   default="~/getcontacts/get_static_contacts.py",
                   help="Full path to get_static_contacts.py on the HPC")
    p.add_argument("--getcontacts-conda-env", default=None,
                   help="Conda environment that has vmd-python installed "
                        "(e.g. 'vmd-python'). When provided, step 4 switches "
                        "to this env before running getcontacts and switches "
                        "back afterward.  If omitted, GETCONTACTS_PYTHON is "
                        "used directly (set via --getcontacts-python).")
    p.add_argument("--getcontacts-python", default=None,
                   help="Python binary used to run getcontacts when NOT using "
                        "--getcontacts-conda-env (e.g. "
                        "~/anaconda3/envs/vmd-python/bin/python). "
                        "Defaults to the active interpreter.")
    p.add_argument("--reduce-path",      default=None,
                   help="Full path to the reduce binary for H-atom addition "
                        "(e.g. ~/anaconda3/envs/reduce/bin/reduce). "
                        "Omit to rely on shutil.which('reduce') or RDKit only.")
    p.add_argument("--out-dir",          type=Path, default=Path("slurm"),
                   help="Output directory for generated scripts")
    p.add_argument("--max-array-size",   type=int, default=10000,
                   help="Maximum number of array indices your cluster allows (--array=0-N). "
                        "Each task will process ceil(N_IDS / max-array-size) PDBs.")
    p.add_argument("--max-array-tasks",  type=int, default=200,
                   help="Suggested cap on simultaneous array tasks (%%N throttle)")
    p.add_argument("--conda-init",       default=None,
                   help="Alias or command required by your HPC to initialise conda "
                        "before 'conda activate' (e.g. 'mycondainit'). "
                        "Run after 'source ~/.bashrc'. Omit if not needed.")
    p.add_argument("--strict",           action="store_true",
                   help="Use afterok dependency in 02_merge_train.sh comment "
                        "(informational only — stages are not auto-chained).")
    return p.parse_args()


def _account_line(account: str | None) -> str:
    return f"#SBATCH --account={account}" if account else ""


def _conda_init_line(conda_init: str | None) -> str:
    """Return the conda-init command (with trailing newline) or empty string."""
    return f"{conda_init}\n        " if conda_init else ""


def write_fetch_ids(args: argparse.Namespace, out_dir: Path) -> None:
    account    = _account_line(args.account)
    conda_init = _conda_init_line(args.conda_init)
    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=vlsbn_fetch
        #SBATCH --partition={args.partition}
        {account}
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=1
        #SBATCH --mem=2G
        #SBATCH --time={args.walltime_fetch}
        #SBATCH --output=slurm/logs/fetch_%j.out
        #SBATCH --error=slurm/logs/fetch_%j.err

        # ---------- environment ----------
        source ~/.bashrc
        {conda_init}conda activate {args.conda_env}

        cd ${{SLURM_SUBMIT_DIR:-$HOME/Project_VLS_BN}}
        mkdir -p slurm/logs data/raw data/work data/raw_contacts data/models data/processed

        # ---------- Stage 0: fetch IDs ----------
        echo "[$(date)] Fetching PDB IDs from RCSB..."
        python scripts/hpc/fetch_ids.py \\
            --out      data/pdb_ids.txt \\
            --out-json data/pdb_ids.json

        if [ $? -ne 0 ]; then echo "fetch_ids.py failed"; exit 1; fi

        N_IDS=$(wc -l < data/pdb_ids.txt)
        MAX_ARRAY={args.max_array_size}
        ACTUAL_TASKS=$(( N_IDS < MAX_ARRAY ? N_IDS : MAX_ARRAY ))
        CHUNK_SIZE=$(( (N_IDS + MAX_ARRAY - 1) / MAX_ARRAY ))
        echo "[$(date)] Done. $N_IDS IDs written to data/pdb_ids.txt"
        echo "  Array tasks needed : $ACTUAL_TASKS  (chunk size: ~$CHUNK_SIZE PDBs/task)"
        echo ""
        echo "Next step — submit the processing array manually:"
        echo "  sbatch --array=0-$((ACTUAL_TASKS - 1))%{args.max_array_tasks} slurm/01_process.sh"
        echo ""
        echo "To stop at a specific step for debugging, set STOP_AFTER:"
        echo "  sbatch --export=ALL,STOP_AFTER=3 --array=0-0 slurm/01_process.sh"
    """).rstrip()

    (out_dir / "00_fetch_ids.sh").write_text(script + "\n")


def write_process(args: argparse.Namespace, out_dir: Path) -> None:
    account    = _account_line(args.account)
    conda_init = _conda_init_line(args.conda_init)

    # Env var exports for the main vlsbn env section
    gc_python_export = (
        f"export GETCONTACTS_PYTHON={args.getcontacts_python}"
        if args.getcontacts_python else ""
    )
    reduce_export = (
        f"export REDUCE_PATH={args.reduce_path}"
        if args.reduce_path else ""
    )

    # Step 4: getcontacts bash section — env switching or direct python path
    if args.getcontacts_conda_env:
        gc_env_activate   = f"conda activate {args.getcontacts_conda_env}"
        gc_python_cmd     = "python"
        gc_env_deactivate = f"conda activate {args.conda_env}"
    else:
        gc_env_activate   = ""
        gc_python_cmd     = "$GETCONTACTS_PYTHON"
        gc_env_deactivate = ""

    # Only emit non-empty lines for env switching
    gc_activate_block   = f"\n        {gc_env_activate}"   if gc_env_activate   else ""
    gc_deactivate_block = f"\n        {gc_env_deactivate}" if gc_env_deactivate else ""

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=vlsbn_proc
        #SBATCH --partition={args.partition}
        {account}
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={args.cpus_process}
        #SBATCH --mem={args.mem_process}
        #SBATCH --time={args.walltime_process}
        #SBATCH --output=slurm/logs/proc_%A_%a.out
        #SBATCH --error=slurm/logs/proc_%A_%a.err
        # NOTE: --array is NOT set here.  Submit with the command printed by 00_fetch_ids.sh,
        # or compute manually:
        #   N=$(wc -l < data/pdb_ids.txt)
        #   TASKS=$(( N < {args.max_array_size} ? N : {args.max_array_size} ))
        #   sbatch --array=0-$((TASKS-1))%{args.max_array_tasks} slurm/01_process.sh
        # Use STOP_AFTER=N (1-6) to halt after a specific step (exits the whole task):
        #   sbatch --export=ALL,STOP_AFTER=3 --array=0-0 slurm/01_process.sh

        # ---------- environment (main: vlsbn) ----------
        source ~/.bashrc
        {conda_init}conda activate {args.conda_env}
        export GETCONTACTS_PATH={args.getcontacts}
        {gc_python_export}
        {reduce_export}

        cd ${{SLURM_SUBMIT_DIR:-$HOME/Project_VLS_BN}}

        TASK_ID=$SLURM_ARRAY_TASK_ID
        STOP_AFTER=${{STOP_AFTER:-6}}

        # Compute chunk: each task owns ceil(N_IDS / {args.max_array_size}) PDBs
        N_IDS=$(wc -l < data/pdb_ids.txt)
        MAX_ARRAY={args.max_array_size}
        CHUNK_SIZE=$(( (N_IDS + MAX_ARRAY - 1) / MAX_ARRAY ))
        START=$(( TASK_ID * CHUNK_SIZE ))

        echo "[$(date)] Task $TASK_ID: lines $START–$((START + CHUNK_SIZE - 1)) of $N_IDS  (chunk=$CHUNK_SIZE, STOP_AFTER=$STOP_AFTER)"

        for ((IDX=START; IDX<START+CHUNK_SIZE; IDX++)); do

        PDB_ID=$(sed -n "$((IDX + 1))p" data/pdb_ids.txt | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
        [ -z "$PDB_ID" ] && continue

        WORK_PDB_DIR=data/work/$PDB_ID
        mkdir -p "$WORK_PDB_DIR"

        echo "[$(date)]   IDX=$IDX  PDB=$PDB_ID"

        # ===== Step 1+2: download, parse, write temp PDBs, delete original =====
        echo "[$(date)]   Step 1+2: download + parse + write temp PDBs"
        python scripts/hpc/step1_prepare.py \\
            --pdb-id  "$PDB_ID" \\
            --work-dir data/work \\
            --raw-dir  data/raw
        RC=$?; if [ $RC -ne 0 ]; then echo "step1_prepare failed for $PDB_ID (rc=$RC)"; continue; fi

        if [ "$STOP_AFTER" -le 2 ]; then
            echo "[$(date)] STOP_AFTER=$STOP_AFTER — halting after step 2."
            exit 0
        fi

        # ===== Step 3: protonate =====
        echo "[$(date)]   Step 3: protonate"
        python scripts/hpc/step2_protonate.py \\
            --pdb-id   "$PDB_ID" \\
            --work-dir data/work
        RC=$?; if [ $RC -ne 0 ]; then echo "step2_protonate failed for $PDB_ID (rc=$RC)"; continue; fi

        if [ "$STOP_AFTER" -le 3 ]; then
            echo "[$(date)] STOP_AFTER=$STOP_AFTER — halting after step 3."
            exit 0
        fi

        # ===== Step 4: getcontacts (explicit bash — runs under getcontacts env) ====={gc_activate_block}
        echo "[$(date)]   Step 4: getcontacts"
        MANIFEST=$WORK_PDB_DIR/manifest.tsv
        if [ ! -f "$MANIFEST" ]; then
            echo "Manifest not found: $MANIFEST — skipping $PDB_ID"; continue
        fi

        while IFS=$'\\t' read -r ligand_id resname n_prot n_lig; do
            [ "$ligand_id" = "ligand_id" ] && continue   # skip header
            GC_INPUT="$WORK_PDB_DIR/${{ligand_id}}_h.pdb"
            [ ! -f "$GC_INPUT" ] && GC_INPUT="$WORK_PDB_DIR/${{ligand_id}}.pdb"
            echo "[$(date)]     $ligand_id (resname=$resname)"
            {gc_python_cmd} {args.getcontacts} \\
                --structure "$GC_INPUT" \\
                --output    "$WORK_PDB_DIR/${{ligand_id}}_contacts.tsv" \\
                --itypes    all \\
                --sele      protein \\
                --sele2     "resname $resname"
        done < "$MANIFEST"{gc_deactivate_block}

        if [ "$STOP_AFTER" -le 4 ]; then
            echo "[$(date)] STOP_AFTER=$STOP_AFTER — halting after step 4."
            exit 0
        fi

        # ===== Step 5: parse getcontacts TSV → raw contacts CSV =====
        echo "[$(date)]   Step 5: parse contacts"
        python scripts/hpc/step3_parse_contacts.py \\
            --pdb-id   "$PDB_ID" \\
            --work-dir data/work \\
            --out-dir  data/raw_contacts
        RC=$?; if [ $RC -ne 0 ]; then echo "step3_parse_contacts failed for $PDB_ID (rc=$RC)"; continue; fi

        if [ "$STOP_AFTER" -le 5 ]; then
            echo "[$(date)] STOP_AFTER=$STOP_AFTER — halting after step 5."
            exit 0
        fi

        # ===== Step 6: featurize → contact-density parquet =====
        echo "[$(date)]   Step 6: featurize"
        python scripts/hpc/step4_featurize.py \\
            --pdb-id       "$PDB_ID" \\
            --work-dir     data/work \\
            --contacts-dir data/raw_contacts \\
            --out-dir      data/features
        RC=$?; if [ $RC -ne 0 ]; then echo "step4_featurize failed for $PDB_ID (rc=$RC)"; continue; fi

        echo "[$(date)]   IDX=$IDX ($PDB_ID) done."

        done  # end PDB chunk loop
        echo "[$(date)] Task $TASK_ID complete."
    """).rstrip()

    (out_dir / "01_process.sh").write_text(script + "\n")


def write_merge_train(args: argparse.Namespace, out_dir: Path) -> None:
    account    = _account_line(args.account)
    conda_init = _conda_init_line(args.conda_init)
    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=vlsbn_merge
        #SBATCH --partition={args.partition}
        {account}
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem={args.mem_merge}
        #SBATCH --time={args.walltime_merge}
        #SBATCH --output=slurm/logs/merge_%j.out
        #SBATCH --error=slurm/logs/merge_%j.err

        # ---------- environment ----------
        source ~/.bashrc
        {conda_init}conda activate {args.conda_env}

        cd ${{SLURM_SUBMIT_DIR:-$HOME/Project_VLS_BN}}
        mkdir -p data/models data/processed

        # ---------- coverage sanity check ----------
        N_ACTUAL=$(ls data/features/*.parquet 2>/dev/null | wc -l)
        N_IDS=$(wc -l < data/pdb_ids.txt 2>/dev/null || echo 0)
        echo "[$(date)] Feature parquets found: $N_ACTUAL  (expected up to $N_IDS PDBs)"
        if [ "$N_ACTUAL" -eq 0 ]; then
            echo "[$(date)] ERROR: no feature parquets in data/features/ — did step 6 run?" >&2
            exit 1
        fi

        echo "[$(date)] Merging feature parquets and training BN..."
        python scripts/hpc/merge_train.py \\
            --features-dir  data/features \\
            --out-features  data/processed/contact_features.parquet \\
            --out-trained   data/models/bn_trained.pkl \\
            --out-reference data/models/bn_reference.pkl

        echo "[$(date)] merge_train.py finished (exit $?)"
    """).rstrip()

    (out_dir / "02_merge_train.sh").write_text(script + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    write_fetch_ids(args, out_dir)
    write_process(args, out_dir)
    write_merge_train(args, out_dir)

    print(f"Generated scripts in {out_dir}/")
    print()
    print("  slurm/00_fetch_ids.sh   — Stage 0: fetch IDs (run first, manually)")
    print("  slurm/01_process.sh     — Stage 1: per-PDB array job (submit manually)")
    print("  slurm/02_merge_train.sh — Stage 2: merge + train BN (submit manually)")
    print()
    print("Workflow:")
    print("  sbatch slurm/00_fetch_ids.sh")
    print("  # wait for it to finish — 00_fetch_ids.sh prints the exact sbatch command, or:")
    print(f"  sbatch --array=0-<min(N_IDS,{args.max_array_size})-1>%{args.max_array_tasks} slurm/01_process.sh")
    print("  # wait for array to finish, then:")
    print("  sbatch slurm/02_merge_train.sh")
    print()
    print("Debug a single PDB (stop after step 3):")
    print("  sbatch --export=ALL,STOP_AFTER=3 --array=0-0 slurm/01_process.sh")
    print()
    print("Settings:")
    print(f"  partition         = {args.partition}")
    print(f"  conda env (main)  = {args.conda_env}")
    print(f"  gc conda env      = {args.getcontacts_conda_env or '(use GETCONTACTS_PYTHON)'}")
    print(f"  max array size    = {args.max_array_size} (cluster limit; ~{args.max_array_size} tasks total)")
    print(f"  max parallel      = {args.max_array_tasks} simultaneous tasks (%N throttle)")
    print(f"  getcontacts       = {args.getcontacts}")
    print(f"  gc python         = {args.getcontacts_python or '(active interpreter)'}")
    print(f"  reduce path       = {args.reduce_path or '(shutil.which or RDKit only)'}")
    print(f"  conda-init        = {args.conda_init or '(none)'}")


if __name__ == "__main__":
    main()

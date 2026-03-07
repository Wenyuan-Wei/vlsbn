#!/usr/bin/env python3
"""
Generate Slurm submission scripts for the VLS_BN HPC pipeline.

Run this script locally (or on the login node) to produce a `slurm/`
directory containing three job scripts plus a master submit script.

The generated pipeline has three stages that chain automatically via
Slurm job dependencies:

    Stage 0  00_fetch_ids.sh   — fetch PDB IDs from RCSB → data/pdb_ids.txt
                                  (also auto-submits stages 1 and 2 on completion)
    Stage 1  01_process.sh     — array job: download → process → delete per batch
    Stage 2  02_merge_train.sh — merge chunk parquets + train BN

Usage
-----
    python scripts/hpc/generate_slurm.py \\
        --partition gpu          \\   # or cpu, short, etc.
        --account   mygroup      \\   # optional, omit if not required
        --conda-env vlsbn        \\
        --batch-size 250         \\
        --walltime-fetch  0:30:00 \\
        --walltime-process 4:00:00 \\
        --walltime-merge   2:00:00 \\
        --mem-process 4G         \\
        --getcontacts ~/getcontacts/get_static_contacts.py

Then on the HPC:
    cd ~/Project_VLS_BN
    sbatch slurm/00_fetch_ids.sh        # stages 1 + 2 submit themselves automatically
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
                   help="Conda environment name")
    p.add_argument("--batch-size",       type=int, default=250,
                   help="Number of PDB IDs each array task processes")
    p.add_argument("--walltime-fetch",   default="0:30:00",
                   help="Walltime for stage 0 (fetch IDs)")
    p.add_argument("--walltime-process", default="4:00:00",
                   help="Walltime per array task (stage 1)")
    p.add_argument("--walltime-merge",   default="2:00:00",
                   help="Walltime for stage 2 (merge + train)")
    p.add_argument("--mem-process",      default="4G",
                   help="Memory per array task")
    p.add_argument("--mem-merge",        default="32G",
                   help="Memory for merge+train job (needs to hold full feature matrix)")
    p.add_argument("--cpus-process",     type=int, default=1,
                   help="CPUs per array task (getcontacts is single-threaded)")
    p.add_argument("--getcontacts",
                   default="~/getcontacts/get_static_contacts.py",
                   help="Full path to get_static_contacts.py on the HPC")
    p.add_argument("--out-dir",          type=Path, default=Path("slurm"),
                   help="Output directory for generated scripts")
    p.add_argument("--max-array-tasks",  type=int, default=2000,
                   help="Hard cap on simultaneous array tasks (cluster courtesy)")
    p.add_argument("--dry-run",          action="store_true",
                   help="Embed --dry-run in 01_process.sh so the array job reports "
                        "its batch plan without downloading or writing any data")
    return p.parse_args()


def _account_line(account: str | None) -> str:
    return f"#SBATCH --account={account}" if account else ""


def write_fetch_ids(args: argparse.Namespace, out_dir: Path) -> None:
    account = _account_line(args.account)
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
        conda activate {args.conda_env}
        export GETCONTACTS_PATH={args.getcontacts}

        cd ${{SLURM_SUBMIT_DIR:-$HOME/Project_VLS_BN}}
        mkdir -p slurm/logs data/chunks data/raw data/models data/processed

        # ---------- Stage 0: fetch IDs ----------
        echo "[$(date)] Fetching PDB IDs…"
        python scripts/hpc/fetch_ids.py --out data/pdb_ids.txt
        if [ $? -ne 0 ]; then echo "fetch_ids.py failed"; exit 1; fi

        N_IDS=$(wc -l < data/pdb_ids.txt)
        BATCH={args.batch_size}
        N_TASKS=$(( (N_IDS + BATCH - 1) / BATCH ))
        LAST=$(( N_TASKS - 1 ))
        echo "[$(date)] $N_IDS IDs → $N_TASKS array tasks (batch=$BATCH)"

        # ---------- Stage 1: submit array job ----------
        JOB1=$(sbatch --parsable \\
            --array=0-${{LAST}}%{args.max_array_tasks} \\
            slurm/01_process.sh)
        echo "[$(date)] Submitted process array: job $JOB1 ($N_TASKS tasks)"

        # ---------- Stage 2: submit merge+train, dependent on array ----------
        JOB2=$(sbatch --parsable \\
            --dependency=afterok:${{JOB1}} \\
            slurm/02_merge_train.sh)
        echo "[$(date)] Submitted merge+train: job $JOB2 (waits for $JOB1)"

        echo "[$(date)] All stages submitted.  Monitor with: squeue -u $USER"
    """).rstrip()

    (out_dir / "00_fetch_ids.sh").write_text(script + "\n")


def write_process(args: argparse.Namespace, out_dir: Path) -> None:
    account = _account_line(args.account)
    dry_run_flag = " \\\n            --dry-run" if args.dry_run else ""
    # Array range is set dynamically by 00_fetch_ids.sh; this template uses
    # a placeholder that sbatch --array overrides at submission time.
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
        # NOTE: --array is NOT set here; 00_fetch_ids.sh sets it dynamically
        #       via "sbatch --array=0-N slurm/01_process.sh"

        # ---------- environment ----------
        source ~/.bashrc
        conda activate {args.conda_env}
        export GETCONTACTS_PATH={args.getcontacts}

        cd ${{SLURM_SUBMIT_DIR:-$HOME/Project_VLS_BN}}

        echo "[$(date)] Task $SLURM_ARRAY_TASK_ID starting on $(hostname)"

        python scripts/hpc/hpc_process.py \\
            --ids-file   data/pdb_ids.txt \\
            --out-dir    data/chunks \\
            --raw-dir    data/raw \\
            --batch-size {args.batch_size}{dry_run_flag}

        echo "[$(date)] Task $SLURM_ARRAY_TASK_ID finished (exit $?)"
    """).rstrip()

    (out_dir / "01_process.sh").write_text(script + "\n")


def write_merge_train(args: argparse.Namespace, out_dir: Path) -> None:
    account = _account_line(args.account)
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
        conda activate {args.conda_env}

        cd ${{SLURM_SUBMIT_DIR:-$HOME/Project_VLS_BN}}
        mkdir -p data/models data/processed

        echo "[$(date)] Merging chunk parquets and training BN…"
        python scripts/hpc/merge_train.py \\
            --chunks-dir    data/chunks \\
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
    print("  slurm/00_fetch_ids.sh   — Stage 0: fetch IDs (auto-submits stages 1+2)")
    print("  slurm/01_process.sh     — Stage 1: array processing job")
    print("  slurm/02_merge_train.sh — Stage 2: merge + train BN")
    print()
    print("To run the full pipeline:")
    print(f"  cd ~/Project_VLS_BN")
    print(f"  sbatch slurm/00_fetch_ids.sh")
    print()
    print("Settings:")
    print(f"  partition    = {args.partition}")
    print(f"  conda env    = {args.conda_env}")
    print(f"  batch size   = {args.batch_size} PDBs/task")
    print(f"  max parallel = {args.max_array_tasks} simultaneous tasks")
    print(f"  getcontacts  = {args.getcontacts}")
    if args.dry_run:
        print()
        print("  *** DRY-RUN MODE: 01_process.sh will report batch plans only,")
        print("      no PDB files will be downloaded or processed. ***")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
HPC array-task script: download → parse → compute contacts → delete.

Each Slurm array task calls this script.  The task reads its exclusive
batch of PDB IDs from a pre-built list, processes them one by one, writes
a per-task chunk parquet, and deletes every PDB/CIF file immediately after
processing to keep disk usage low.

Progress is tracked in per-task "done" files so interrupted tasks can
resume without reprocessing completed structures.

Usage (Slurm sets SLURM_ARRAY_TASK_ID automatically)
-----
    python scripts/hpc/hpc_process.py \\
        --ids-file  data/pdb_ids.txt \\
        --out-dir   data/chunks \\
        --raw-dir   data/raw \\
        --batch-size 250

    # Manual test of task 3:
    python scripts/hpc/hpc_process.py --ids-file data/pdb_ids.txt \\
        --out-dir data/chunks --task-id 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlsbn.pipeline.contacts import compute_features
from vlsbn.pipeline.fetch import download_pdb
from vlsbn.pipeline.parse import parse_pdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_META_COLS = {"pdb_id", "ligand_id"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLS_BN HPC array task")
    p.add_argument("--ids-file",   type=Path, required=True,
                   help="Text file with one PDB ID per line (from fetch_ids.py)")
    p.add_argument("--out-dir",    type=Path, default=Path("data/chunks"),
                   help="Directory for chunk parquets and done-tracking files")
    p.add_argument("--raw-dir",    type=Path, default=Path("data/raw"),
                   help="Temporary directory for downloaded PDB files")
    p.add_argument("--batch-size", type=int,  default=250,
                   help="Number of PDB IDs per array task")
    p.add_argument("--task-id",    type=int,  default=None,
                   help="Override $SLURM_ARRAY_TASK_ID (useful for local testing)")
    p.add_argument("--delay",      type=float, default=0.05,
                   help="Per-download sleep in seconds (rate-limiting)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Determine task ID and batch                                          #
    # ------------------------------------------------------------------ #
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    all_ids = [l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()]
    start   = task_id * args.batch_size
    batch   = all_ids[start : start + args.batch_size]

    logger.info(
        "Task %d: IDs %d–%d  (%d total)", task_id, start, start + len(batch) - 1, len(batch)
    )

    if not batch:
        logger.info("No IDs for this task. Exiting.")
        return

    # ------------------------------------------------------------------ #
    # Resume: read already-processed IDs for this task                    #
    # ------------------------------------------------------------------ #
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    done_file  = args.out_dir / f"done_{task_id:05d}.txt"
    chunk_file = args.out_dir / f"chunk_{task_id:05d}.parquet"

    done_ids: set[str] = set()
    if done_file.exists():
        done_ids = {l.strip() for l in done_file.read_text().splitlines() if l.strip()}
        logger.info("Resuming: %d IDs already done, skipping.", len(done_ids))

    # Load rows from a pre-existing partial chunk (in case we crashed mid-write)
    rows: list[dict] = []
    if chunk_file.exists() and done_ids:
        rows = pd.read_parquet(chunk_file).to_dict("records")
        logger.info("Loaded %d existing rows from %s.", len(rows), chunk_file)

    # ------------------------------------------------------------------ #
    # Process each PDB ID                                                  #
    # ------------------------------------------------------------------ #
    n_complexes = 0
    n_failed    = 0

    with done_file.open("a") as done_fh:
        for pdb_id in batch:
            if pdb_id in done_ids:
                continue

            pdb_path: Path | None = None
            try:
                pdb_path = download_pdb(pdb_id, dest_dir=args.raw_dir, delay=args.delay)
                complexes = parse_pdb(pdb_path)

                for c in complexes:
                    feats = compute_features(c)
                    feats["pdb_id"]    = c.pdb_id
                    feats["ligand_id"] = c.ligand_id
                    rows.append(feats)
                    n_complexes += 1

            except Exception as exc:
                logger.warning("FAILED %s: %s", pdb_id, exc)
                n_failed += 1

            finally:
                # Delete PDB immediately after processing to save disk space
                if pdb_path is not None and pdb_path.exists():
                    pdb_path.unlink()

            # Record as done and flush so progress survives a crash
            done_fh.write(pdb_id + "\n")
            done_fh.flush()

    # ------------------------------------------------------------------ #
    # Write chunk parquet                                                  #
    # ------------------------------------------------------------------ #
    if rows:
        df = pd.DataFrame(rows).fillna(0.0)
        meta = [c for c in ("pdb_id", "ligand_id") if c in df.columns]
        feat = [c for c in df.columns if c not in set(meta)]
        df   = df[meta + feat]
        df.to_parquet(chunk_file, index=False)
        logger.info(
            "Task %d done: %d complexes, %d failed → %s",
            task_id, n_complexes, n_failed, chunk_file,
        )
    else:
        logger.warning("Task %d: no complexes extracted.", task_id)


if __name__ == "__main__":
    main()

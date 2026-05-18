"""
Sequential launcher for the BSPS6 100-cell grid.

Usage from repo root:
    python src/run_grid.py [--dry_run] [--grid main|h5a|h5b|all]
                               [--retry_all] [--max_runs N] [--seed 42]

The launcher:
  1. Builds the grid from the locked architectures × encodings × datasets
     (defined in src/config/architectures.py and src/config/constants.py).
  2. Reads runs.csv to find cells with status='completed'.
  3. Skips completed cells (unless --retry_all is passed).
  4. Runs the remaining cells sequentially via subprocess to `src/main.py`.

The grid is 100 runs at 1 seed:
  * Main grid: 4 archs × 4 encodings × 5 datasets × 1 seed                = 80
  * H5a:       SH at ⌈L_95⌉ for ERA5 (L_max=13) and HDRI_sky (L_max=31)   =  8
  * H5b:       SH at L_max=16 for ETOPO1, HDRI_urban, CMB                 = 12

Resume policy: a cell's row in runs.csv with status='completed' will cause
that cell to be skipped on the next launcher invocation. Failed runs
(status='oom', 'error_*', etc.) are retried. To force a full rerun of every
cell — including completed ones — pass --retry_all.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from config.architectures import ARCHITECTURES, architecture_cli_args        # noqa: E402
from config.constants import CE_CHOICES, DATASET_CHOICES                     # noqa: E402


# ----- H5a / H5b specifications (preregistration §3.5) ---------------------
# H5a: SH at ⌈L_95⌉ for low-bandwidth datasets, compared against main-grid
#      L_max=32 cells.
H5A_RUNS = (
    {'dataset': 'era5',     'sh_lmax': 13},     # L_95 = 13
    {'dataset': 'hdri_sky', 'sh_lmax': 31},     # L_95 = 31
)
# H5b: SH at L_max=16 for high-bandwidth datasets, compared against main-grid
#      L_max=32 cells.
H5B_DATASETS = ('etopo1', 'hdri_urban', 'cmb')
H5B_LMAX = 16


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def build_main_grid(seed: int) -> list[dict]:
    """The 80 main-grid cells: 4 archs × 4 encodings × 5 datasets × 1 seed.
    Encoding kwargs use the opts.py defaults (locked in preregistration §3.3)."""
    cells: list[dict] = []
    for arch_key in ARCHITECTURES:
        for ce in CE_CHOICES:
            for dataset in DATASET_CHOICES:
                cells.append({
                    'arch_key': arch_key,
                    'ce':       ce,
                    'dataset':  dataset,
                    'seed':     seed,
                    'extra_cli': [],
                    'tag':      'main',
                })
    return cells


def build_h5a_grid(seed: int) -> list[dict]:
    """8 H5a cells: 4 archs × 2 datasets × 1 matched-L_max per dataset."""
    cells: list[dict] = []
    for arch_key in ARCHITECTURES:
        for run in H5A_RUNS:
            cells.append({
                'arch_key': arch_key,
                'ce':       'spherical-harmonics',
                'dataset':  run['dataset'],
                'seed':     seed,
                'extra_cli': ['--sh_lmax', str(run['sh_lmax'])],
                'tag':      f"h5a_lmax{run['sh_lmax']}",
            })
    return cells


def build_h5b_grid(seed: int) -> list[dict]:
    """12 H5b cells: 4 archs × 3 high-bandwidth datasets × L_max=16."""
    cells: list[dict] = []
    for arch_key in ARCHITECTURES:
        for dataset in H5B_DATASETS:
            cells.append({
                'arch_key': arch_key,
                'ce':       'spherical-harmonics',
                'dataset':  dataset,
                'seed':     seed,
                'extra_cli': ['--sh_lmax', str(H5B_LMAX)],
                'tag':      f'h5b_lmax{H5B_LMAX}',
            })
    return cells


# ---------------------------------------------------------------------------
# Cell-identity tuple — matches the columns RunsCSVLogger writes.
# ---------------------------------------------------------------------------
def cell_key_from_plan(cell: dict) -> tuple:
    """Canonical identity tuple for a planned cell. Must match
    cell_key_from_row(row) for the same logical cell."""
    arch_cfg = ARCHITECTURES[cell['arch_key']]
    ce = cell['ce']

    # Parse --flag value pairs back into a dict so we can derive
    # encoding_kwargs the same way opts.py's _encoding_kwargs_from_hparams does.
    extra_dict: dict[str, str] = {}
    extra_cli = cell['extra_cli']
    for i in range(0, len(extra_cli), 2):
        flag = extra_cli[i].lstrip('-')
        extra_dict[flag] = extra_cli[i + 1]

    if ce == 'spherical-harmonics':
        ce_kwargs = {'L_max': int(extra_dict.get('sh_lmax', 32))}
    elif ce == 'spherical-rff':
        ce_kwargs = {
            'num_features': int(extra_dict.get('rff_num_features', 32)),
            'sigma':        float(extra_dict.get('rff_sigma', 10.0)),
            'seed':         int(extra_dict.get('rff_seed', 42)),
        }
    else:
        ce_kwargs = {}

    ce_json = json.dumps(ce_kwargs, sort_keys=True, default=str)
    return (
        cell['dataset'],
        ce,
        arch_cfg.get('arch', ''),
        arch_cfg.get('act', ''),
        arch_cfg.get('mlp_act', ''),
        arch_cfg.get('kan_act', ''),
        arch_cfg.get('pe', ''),
        int(cell['seed']),
        ce_json,
    )


def cell_key_from_row(row: dict) -> tuple:
    """Canonical identity tuple from a CSV row written by RunsCSVLogger."""
    return (
        row.get('dataset', ''),
        row.get('ce', ''),
        row.get('arch', ''),
        row.get('act', ''),
        row.get('mlp_act', ''),
        row.get('kan_act', ''),
        row.get('pe', ''),
        int(row.get('seed', 0) or 0),
        row.get('encoding_kwargs_json', '{}'),
    )


def load_completed_cells(runs_csv: str) -> set[tuple]:
    """Cells with status='completed' in runs.csv. Empty set if file is absent."""
    completed: set[tuple] = set()
    if not os.path.exists(runs_csv):
        return completed
    with open(runs_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'completed':
                completed.add(cell_key_from_row(row))
    return completed


# ---------------------------------------------------------------------------
# Per-cell launch
# ---------------------------------------------------------------------------
def cell_save_dir(log_root: str, cell: dict) -> str:
    """Per-cell save directory so TB logs don't pile up under version_N."""
    parts = [cell['arch_key'], cell['ce'], cell['dataset'],
             f"seed{cell['seed']}"]
    if cell['tag'] != 'main':
        parts.append(cell['tag'])
    return os.path.join(log_root, '__'.join(parts))


def describe_cell(cell: dict) -> str:
    extras = ' '.join(cell['extra_cli'])
    base = (f"{cell['arch_key']:18s} / {cell['ce']:22s} / "
            f"{cell['dataset']:12s} / seed={cell['seed']}")
    return f"{base} {extras}" if extras else base


def run_cell(cell: dict, log_root: str, runs_csv: str) -> int:
    """Spawn `python src/main.py` with the cell's CLI args. Child stdout/stderr
    flow through to the parent so the user sees TB progress bars live.
    Returns the subprocess returncode (0 on success)."""
    save_dir = cell_save_dir(log_root, cell)
    os.makedirs(save_dir, exist_ok=True)

    args = [
        sys.executable, '-u', 'main.py',
        '--dataset', cell['dataset'],
        '--ce',      cell['ce'],
        '--seed',    str(cell['seed']),
        '--save_dir', save_dir,
        '--runs_csv', runs_csv,
        *architecture_cli_args(cell['arch_key']),
        *cell['extra_cli'],
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = env.get('PYTHONPATH', '')

    print(f"  $ {' '.join(args)}")
    result = subprocess.run(args, env=env, check=False)
    return result.returncode


def format_duration(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--runs_csv', default='results/runs.csv',
                        help='Path to runs.csv (default: results/runs.csv).')
    parser.add_argument('--log_root', default='logs/grid',
                        help='Root dir for per-cell TB log dirs.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for all cells (preregistration §3.5; default 42).')
    parser.add_argument('--grid', choices=['main', 'h5a', 'h5b', 'all'],
                        default='all',
                        help='Which subset of the grid to run.')
    parser.add_argument('--retry_all', action='store_true',
                        help="Ignore existing runs.csv; rerun every cell.")
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the plan; do not actually launch any cell.')
    parser.add_argument('--max_runs', type=int, default=0,
                        help='Cap on number of pending cells to run '
                             '(0 = unlimited).')
    parser.add_argument('--stop_on_failure', action='store_true',
                        help='Abort the launcher on the first failing cell '
                             '(default: continue past failures).')
    args = parser.parse_args()

    # ---- Build grid ----
    cells: list[dict] = []
    if args.grid in ('main', 'all'):
        cells.extend(build_main_grid(args.seed))
    if args.grid in ('h5a', 'all'):
        cells.extend(build_h5a_grid(args.seed))
    if args.grid in ('h5b', 'all'):
        cells.extend(build_h5b_grid(args.seed))

    # ---- Resume detection ----
    completed = set() if args.retry_all else load_completed_cells(args.runs_csv)
    pending = [c for c in cells if cell_key_from_plan(c) not in completed]
    if args.max_runs > 0:
        pending = pending[:args.max_runs]

    # ---- Plan summary ----
    print('=' * 72)
    print(f'BSPS6 grid launcher (preregistration §3.5)')
    print('=' * 72)
    print(f"Grid subset : {args.grid}")
    print(f"Total cells : {len(cells)}")
    print(f"Completed   : {len(completed)} (from {args.runs_csv})")
    print(f"Pending     : {len(pending)}"
          + (f' (capped via --max_runs={args.max_runs})'
             if args.max_runs > 0 and args.max_runs < len(cells) - len(completed)
             else ''))
    print(f"Save dir    : {args.log_root}")
    print(f"Runs CSV    : {args.runs_csv}")
    print(f"Seed        : {args.seed}")
    print(f"Retry all   : {args.retry_all}")
    print(f"Stop on fail: {args.stop_on_failure}")
    print('=' * 72)

    if args.dry_run:
        print('\nDry-run plan (no cells will be launched):\n')
        for i, c in enumerate(pending, 1):
            print(f"  [{i:3d}/{len(pending)}] {describe_cell(c)}")
        return

    if not pending:
        print("\nNothing to do — every requested cell already has "
              "status='completed' in runs.csv.")
        print("(Use --retry_all to force a full re-run.)")
        return

    # ---- Execute pending cells sequentially ----
    start = time.time()
    failures: list[tuple[dict, int]] = []
    ran = 0
    for i, cell in enumerate(pending, 1):
        elapsed = time.time() - start
        eta_str = ''
        if ran > 0:
            per_cell = elapsed / ran
            remaining = per_cell * (len(pending) - i + 1)
            eta_str = f"  (avg {format_duration(per_cell)}/cell, ETA {format_duration(remaining)})"

        print(f"\n[{i:3d}/{len(pending)}] {describe_cell(cell)}{eta_str}")
        rc = run_cell(cell, args.log_root, args.runs_csv)
        ran += 1
        if rc != 0:
            failures.append((cell, rc))
            print(f"  ! cell failed: returncode={rc}")
            if args.stop_on_failure:
                print('  Aborting due to --stop_on_failure.')
                break

    # ---- Final summary ----
    total = time.time() - start
    print('\n' + '=' * 72)
    print(f"Done. Ran {ran - len(failures)}/{ran} cells successfully in "
          f"{format_duration(total)}.")
    if failures:
        print(f"\n{len(failures)} failures (consult runs.csv "
              f"and {args.log_root}/<cell>/ for details):")
        for cell, rc in failures:
            print(f"  rc={rc}: {describe_cell(cell)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

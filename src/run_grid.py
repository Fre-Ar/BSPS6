"""
Sequential launcher for the BSPS6 grid.

Usage from repo root:
    python src/run_grid.py [--dry_run] [--grid GRID] [--retry_all]
                               [--max_runs N] [--seed 42]
                               [--filter_pe LIST] [--filter_act LIST]
                               [--filter_dataset LIST]
The launcher:
  1. Builds the grid from the locked (activation × PE) cells defined in
     src/config/architectures.py, crossed with the 5 datasets in
     src/config/constants.py.
  2. Reads runs.csv to find cells with status='completed'.
  3. Skips completed cells (unless --retry_all is passed).
  4. Optionally filters the remaining set by PE / activation / dataset.
  5. Runs the remaining cells sequentially via subprocess to `src/main.py`.

At 1 seed:
  * Main grid:        every (activation × PE) cell × every dataset
  * SH post-saturation sub-grid: SH at ⌈L_95⌉ for each low-bandwidth dataset
                                 (ERA5: L_max=13, HDRI_sky: L_max=31),
                                 one cell per activation
  * SH pre-saturation sub-grid:  SH at L_max=16 for each high-bandwidth dataset
                                 (ETOPO1, HDRI_urban, CMB), one cell per
                                 activation

`--grid` choices (which sub-grids to include):
  * `main`               — main grid only
  * `sh_post_saturation` — SH ⌈L_95⌉ sub-grid only
  * `sh_pre_saturation`  — SH L_max=16 sub-grid only
  * `sh_ablation`        — both SH sub-grids (no main grid)
  * `all`                — everything (default)

`--filter_*` choices narrow the selected set further. Useful when running
a subset on a borrowed machine and copying the resulting runs.csv +
checkpoints back. Examples:

    # Run just the SH cells on CMB on a friend's PC, then copy back.
    python src/run_grid.py --filter_pe sh --filter_dataset cmb

    # Run all of `relu` and `scaled_sine` cells across every PE / dataset.
    python src/run_grid.py --filter_act relu,scaled_sine

    # Run only the SH-ablation sub-grids on a borrowed Nvidia box.
    python src/run_grid.py --grid sh_ablation

Resume policy: a cell's row in runs.csv with status='completed' will cause
that cell to be skipped on the next launcher invocation. Failed runs
(status='oom', 'error_*', etc.) are retried. `--retry_all` forces a full
re-run of every cell, including completed ones.
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

from config.architectures import (                                          # noqa: E402
    ACTIVATIONS, PE_CELLS, cell_keys, cell_config, cell_cli_args,
)
from config.constants import DATASET_CHOICES                     # noqa: E402


# ----- SH ablation sub-grid specifications (preregistration §2.4) ----------
# The SH cells (one per activation). The SH ablation re-runs only the cells
# that *use* SH — other PE cells are unaffected.
SH_CELL_KEYS: tuple[str, ...] = tuple(
    f'{act_key}__sh' for act_key in ACTIVATIONS
)

# Post-saturation: SH at ⌈L_95⌉ for low-bandwidth datasets, compared against
# main-grid L_max=32 cells.
SH_POST_SATURATION_RUNS = (
    {'dataset': 'era5',     'sh_lmax': 13},   # L_95 = 13
    {'dataset': 'hdri_sky', 'sh_lmax': 31},   # L_95 = 31
)

# Pre-saturation: SH at L_max=16 for high-bandwidth datasets, compared
# against main-grid L_max=32 cells.
SH_PRE_SATURATION_DATASETS = ('etopo1', 'hdri_urban', 'cmb')
SH_PRE_SATURATION_LMAX = 16


_GRID_CHOICES = (
    'main',
    'sh_post_saturation',
    'sh_pre_saturation',
    'sh_ablation',          # both SH sub-grids
    'all',                  # main + both SH sub-grids
)



# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def build_main_grid(seed: int) -> list[dict]:
    """Main grid: every cell × every dataset, all at the given seed.
    Encoding kwargs use the opts.py defaults."""
    cells: list[dict] = []
    for cell_key in cell_keys():
        for dataset in DATASET_CHOICES:
            cells.append({
                'cell_key': cell_key,
                'dataset':  dataset,
                'seed':     seed,
                'extra_cli': [],
                'tag':      'main',
            })
    return cells


def build_sh_post_saturation_grid(seed: int) -> list[dict]:
    """Every SH cell × every low-bandwidth dataset, at the dataset's
    matched L_max = ⌈L_95⌉."""
    cells: list[dict] = []
    for cell_key in SH_CELL_KEYS:
        for run in SH_POST_SATURATION_RUNS:
            cells.append({
                'cell_key': cell_key,
                'dataset':  run['dataset'],
                'seed':     seed,
                'extra_cli': ['--sh_lmax', str(run['sh_lmax'])],
                'tag':      f"sh_post_lmax{run['sh_lmax']}",
            })
    return cells


def build_sh_pre_saturation_grid(seed: int) -> list[dict]:
    """Every SH cell × every high-bandwidth dataset, at L_max=16."""
    cells: list[dict] = []
    for cell_key in SH_CELL_KEYS:
        for dataset in SH_PRE_SATURATION_DATASETS:
            cells.append({
                'cell_key': cell_key,
                'dataset':  dataset,
                'seed':     seed,
                'extra_cli': ['--sh_lmax', str(SH_PRE_SATURATION_LMAX)],
                'tag':      f'sh_pre_lmax{SH_PRE_SATURATION_LMAX}',
            })
    return cells

def build_grid(grid_subset: str, seed: int) -> list[dict]:
    """Build the union of sub-grids requested by `grid_subset`."""
    if grid_subset not in _GRID_CHOICES:
        raise ValueError(
            f"Unknown --grid '{grid_subset}'. Available: {_GRID_CHOICES}"
        )
    cells: list[dict] = []
    if grid_subset in ('main', 'all'):
        cells.extend(build_main_grid(seed))
    if grid_subset in ('sh_post_saturation', 'sh_ablation', 'all'):
        cells.extend(build_sh_post_saturation_grid(seed))
    if grid_subset in ('sh_pre_saturation', 'sh_ablation', 'all'):
        cells.extend(build_sh_pre_saturation_grid(seed))
    return cells

# ---------------------------------------------------------------------------
# Cell-identity tuple — matches the columns RunsCSVLogger writes.
# ---------------------------------------------------------------------------
def cell_key_from_plan(cell: dict) -> tuple:
    """Canonical identity tuple for a planned cell. Must match
    cell_key_from_row(row) for the same logical cell.

    Tuple shape:
      (dataset, ce, act, pe, seed, encoding_kwargs_json)
    """
    cfg = cell_config(cell['cell_key'])
    ce = cfg.get('ce', '')

    # Parse `extra_cli` (a flat list ['--flag', value, ...]) into a dict so
    # we can derive encoding_kwargs the same way opts.py's
    # _encoding_kwargs_from_hparams does.
    extra_dict: dict[str, str] = {}
    extra_cli = cell['extra_cli']
    for i in range(0, len(extra_cli), 2):
        flag = extra_cli[i].lstrip('-')
        extra_dict[flag] = extra_cli[i + 1]

    if ce == 'spherical-harmonics':
        ce_kwargs = {'L_max': int(extra_dict.get('sh_lmax', cfg.get('sh_lmax', 32)))}
    else:
        ce_kwargs = {}

    ce_json = json.dumps(ce_kwargs, sort_keys=True, default=str)
    return (
        cell['dataset'],
        ce,
        cfg.get('act', ''),
        cfg.get('pe', ''),
        int(cell['seed']),
        ce_json,
    )


def cell_key_from_row(row: dict) -> tuple:
    """Canonical identity tuple from a CSV row written by RunsCSVLogger."""
    return (
        row.get('dataset', ''),
        row.get('ce', ''),
        row.get('act', ''),
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
# Filtering
# ---------------------------------------------------------------------------
def _parse_filter_list(arg: str | None) -> set[str] | None:
    """Parse a comma-separated CLI filter argument into a set, or None to
    indicate 'no filter'."""
    if arg is None or arg.strip() == '':
        return None
    return {tok.strip() for tok in arg.split(',') if tok.strip()}


def _activation_from_cell_key(cell_key: str) -> str:
    """Extract the activation token from a cell_key (cell_keys are
    `<activation>__<pe>`)."""
    return cell_key.split('__', 1)[0]


def _pe_from_cell_key(cell_key: str) -> str:
    """Extract the PE token from a cell_key."""
    return cell_key.split('__', 1)[1]


def apply_filters(
    cells: list[dict],
    pe_filter: set[str] | None,
    act_filter: set[str] | None,
    dataset_filter: set[str] | None,
) -> list[dict]:
    """Narrow a built grid down to cells matching ALL provided filters.
    A None filter is a no-op for that dimension."""
    out: list[dict] = []
    for c in cells:
        if pe_filter is not None and _pe_from_cell_key(c['cell_key']) not in pe_filter:
            continue
        if act_filter is not None and _activation_from_cell_key(c['cell_key']) not in act_filter:
            continue
        if dataset_filter is not None and c['dataset'] not in dataset_filter:
            continue
        out.append(c)
    return out


def _validate_filter(
    name: str, values: set[str] | None, available: set[str],
) -> None:
    """Fail loudly if a filter lists tokens we don't recognize."""
    if values is None:
        return
    unknown = sorted(values - available)
    if unknown:
        raise SystemExit(
            f"[run_grid] --filter_{name} contains unknown tokens "
            f"{unknown}. Available: {sorted(available)}."
        )


# ---------------------------------------------------------------------------
# Per-cell launch
# ---------------------------------------------------------------------------
def cell_save_dir(log_root: str, cell: dict) -> str:
    """Per-cell save directory so TB logs don't pile up under version_N."""
    parts = [cell['cell_key'], cell['dataset'], f"seed{cell['seed']}"]
    if cell['tag'] != 'main':
        parts.append(cell['tag'])
    return os.path.join(log_root, '__'.join(parts))


def describe_cell(cell: dict) -> str:
    extras = ' '.join(cell['extra_cli'])
    tag = f" [{cell['tag']}]" if cell['tag'] != 'main' else ''
    base = (f"{cell['cell_key']:30s} / {cell['dataset']:12s} / "
            f"seed={cell['seed']}{tag}")
    return f"{base} {extras}" if extras else base


def run_cell(cell: dict, log_root: str, runs_csv: str) -> int:
    """Spawn `python src/main.py` with the cell's CLI args. Child stdout/stderr
    flow through to the parent so the user sees TB progress bars live.
    Returns the subprocess returncode (0 on success)."""
    save_dir = cell_save_dir(log_root, cell)
    os.makedirs(save_dir, exist_ok=True)

    args = [
        sys.executable, '-u', 'src/main.py',
        '--dataset', cell['dataset'],
        '--seed',    str(cell['seed']),
        '--save_dir', save_dir,
        '--runs_csv', runs_csv,
        *cell_cli_args(cell['cell_key']),
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
    parser.add_argument('--grid', choices=_GRID_CHOICES, default='all',
                        help='Which sub-grids to include (default: all).')
    parser.add_argument('--filter_pe', default=None,
                        help='Comma-separated list of PE cells to keep '
                             '(e.g. "sh,fkan"). Default: all PE cells.')
    parser.add_argument('--filter_act', default=None,
                        help='Comma-separated list of activations to keep '
                             '(e.g. "relu,scaled_sine"). Default: all activations.')
    parser.add_argument('--filter_dataset', default=None,
                        help='Comma-separated list of datasets to keep '
                             '(e.g. "etopo1,cmb"). Default: all datasets.')
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

    # ---- Build + filter grid ----
    cells = build_grid(args.grid, seed=args.seed)
    pe_filter      = _parse_filter_list(args.filter_pe)
    act_filter     = _parse_filter_list(args.filter_act)
    dataset_filter = _parse_filter_list(args.filter_dataset)
    _validate_filter('pe',      pe_filter,      set(PE_CELLS.keys()))
    _validate_filter('act',     act_filter,     set(ACTIVATIONS.keys()))
    _validate_filter('dataset', dataset_filter, set(DATASET_CHOICES))
    cells = apply_filters(cells, pe_filter, act_filter, dataset_filter)

    # ---- Resume detection ----
    completed = set() if args.retry_all else load_completed_cells(args.runs_csv)
    pending = [c for c in cells if cell_key_from_plan(c) not in completed]
    if args.max_runs > 0:
        pending = pending[:args.max_runs]

    # ---- Plan summary ----
    filters_str = ', '.join(filter(None, [
        f'pe={sorted(pe_filter)}'      if pe_filter      else None,
        f'act={sorted(act_filter)}'    if act_filter     else None,
        f'dataset={sorted(dataset_filter)}' if dataset_filter else None,
    ])) or 'none'
    
    print('=' * 72)
    print('BSPS6 grid launcher')
    print('=' * 72)
    print(f"Grid subset : {args.grid}")
    print(f"Filters     : {filters_str}")
    print(f"Selected    : {len(cells)} cells")
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
        print("\nNothing to do — every selected cell already has "
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
            eta_str = (f"  (avg {format_duration(per_cell)}/cell, "
                       f"ETA {format_duration(remaining)})")

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

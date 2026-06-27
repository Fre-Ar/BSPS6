"""
Unit tests for src/run_grid.py — grid construction, cell-key canonicality,
and resume-detection against a synthetic runs.csv (preregistration §3.5).

We do NOT spawn any subprocesses; all assertions are against the grid-building
and key-derivation logic.

Run from repo root:
    PYTHONPATH=src python tests/test_run_grid.py
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

# Import the launcher as a module (it's set up to be importable).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'run_grid', os.path.join(REPO_ROOT, 'src', 'run_grid.py'),
)
run_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_grid)

from config.architectures import (                                       # noqa: E402
    cell_keys, cell_config, ACTIVATIONS, PE_CELLS
)
from config.constants import DATASET_CHOICES                              # noqa: E402

# Minimal column subset for synthetic CSVs: only the columns
# `cell_key_from_row` reads, plus `status` for resume filtering.
# (The real `RUNS_CSV_SCHEMA` in callbacks/runs_csv.py is much larger but
# requires torch via PL's Callback base class — keeping the test torch-free.)
_TEST_CSV_COLUMNS = (
    'dataset', 'ce', 'act', 'pe', 'seed', 'encoding_kwargs_json', 'status',
)

# Expected grid sizes, derived from the (activation × PE) cross product and
# the sub-grids built by run_grid.build_sh_post_saturation_grid / build_sh_pre_saturation_grid. 
EXPECTED_N_CELLS = len(ACTIVATIONS) * len(PE_CELLS)
EXPECTED_MAIN = EXPECTED_N_CELLS * len(DATASET_CHOICES)
EXPECTED_POST_SATURATION = len(ACTIVATIONS) * 2
EXPECTED_PRE_SATURATION = len(ACTIVATIONS) * 3
EXPECTED_TOTAL = EXPECTED_MAIN + EXPECTED_POST_SATURATION + EXPECTED_PRE_SATURATION


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def test_main_grid_size_and_uniqueness() -> None:
    """Main grid = (cells × datasets), every identity tuple unique."""
    
    print('\n[run_grid] main grid: shape & uniqueness ...')
    cells = run_grid.build_main_grid(seed=42)
    assert len(cells) == EXPECTED_MAIN, (
        f'expected {EXPECTED_MAIN} cells, got {len(cells)}'
    )
    keys = {run_grid.cell_key_from_plan(c) for c in cells}
    assert len(keys) == len(cells), 'duplicate cell-identity tuples in main grid'
    print(f'  OK {EXPECTED_MAIN} cells unique '
          f'({EXPECTED_N_CELLS} cells × {len(DATASET_CHOICES)} datasets).')


def test_sh_post_saturation_grid() -> None:
    
    print('\n[run_grid] sh_post_saturation grid: matched L_max per dataset ...')
    cells = run_grid.build_sh_post_saturation_grid(seed=42)
    assert len(cells) == EXPECTED_POST_SATURATION, f'expected {EXPECTED_POST_SATURATION}, got {len(cells)}'
    for c in cells:
        assert c['cell_key'].endswith('__sh'), c
    by_dataset: dict[str, set[int]] = {}
    for c in cells:
        lmax_idx = c['extra_cli'].index('--sh_lmax') + 1
        by_dataset.setdefault(c['dataset'], set()).add(int(c['extra_cli'][lmax_idx]))
    assert by_dataset['era5'] == {13}, f'ERA5 L_max: {by_dataset["era5"]}'
    assert by_dataset['hdri_sky'] == {31}, f'HDRI_sky L_max: {by_dataset["hdri_sky"]}'
    print(f'  OK sh_post_saturation: ERA5 L_max=13, HDRI_sky L_max=31, {len(ACTIVATIONS)} '
          f'activations each.')


def test_sh_pre_saturation_grid() -> None:
    """sh_pre_saturation runs every SH cell on each of the high-bandwidth datasets at L_max=16."""
    print('\n[run_grid] sh_pre_saturation grid: L_max=16 ...')
    cells = run_grid.build_sh_pre_saturation_grid(seed=42)
    assert len(cells) == EXPECTED_PRE_SATURATION, f'expected {EXPECTED_PRE_SATURATION}, got {len(cells)}'
    for c in cells:
        assert c['cell_key'].endswith('__sh'), c
        lmax_idx = c['extra_cli'].index('--sh_lmax') + 1
        assert int(c['extra_cli'][lmax_idx]) == 16
    datasets = {c['dataset'] for c in cells}
    assert datasets == {'etopo1', 'hdri_urban', 'cmb'}, datasets
    print(f'  OK sh_pre_saturation: L_max=16 for ETOPO1, HDRI_urban, CMB, '
          f'{len(ACTIVATIONS)} activations each.')



def test_full_grid_composes_subsets() -> None:
    """The full grid is the disjoint union of the main grid and the SH
    sub-grids; the sizes add up exactly (no overlap, no missing cells)."""
    print('\n[run_grid] full grid composition ...')
    n = (len(run_grid.build_main_grid(42))
         + len(run_grid.build_sh_post_saturation_grid(42))
         + len(run_grid.build_sh_pre_saturation_grid(42)))
    assert n == EXPECTED_TOTAL, f'expected {EXPECTED_TOTAL}, got {n}'
    # build_grid('all') / 'main' / 'sh_ablation' return the right cell counts.
    assert len(run_grid.build_grid('all', 42)) == n
    assert len(run_grid.build_grid('main', 42)) == EXPECTED_MAIN
    assert len(run_grid.build_grid('sh_ablation', 42)) == (
        EXPECTED_POST_SATURATION + EXPECTED_PRE_SATURATION
    )
    print(f'  OK {EXPECTED_MAIN} (main) + {EXPECTED_POST_SATURATION} (sh_post) + '
          f'{EXPECTED_PRE_SATURATION} (sh_pre) = {n} runs.')



def test_filters_narrow_the_grid() -> None:
    """--filter_pe / --filter_act / --filter_dataset narrow the selected set
    correctly (and combine as set intersection)."""
    print('\n[run_grid] filter helpers ...')
    cells = run_grid.build_main_grid(42)
    # Single-axis: keep only the SH cells.
    only_sh = run_grid.apply_filters(cells, {'sh'}, None, None)
    assert all(c['cell_key'].endswith('__sh') for c in only_sh)
    assert len(only_sh) == len(ACTIVATIONS) * len(DATASET_CHOICES)
    # Multi-axis intersection: 2 PEs × 1 activation × 1 dataset = 2 cells.
    sub = run_grid.apply_filters(
        cells, {'none_angular', 'rff'}, {'relu'}, {'cmb'},
    )
    assert len(sub) == 2
    for c in sub:
        assert c['dataset'] == 'cmb'
        act, pe = c['cell_key'].split('__', 1)
        assert act == 'relu' and pe in {'none_angular', 'rff'}
    # None-filter is a no-op.
    assert run_grid.apply_filters(cells, None, None, None) == cells
    print(f'  OK pe={{sh}} → {len(only_sh)} cells; '
          f'pe={{none_angular,rff}} × act={{relu}} × dataset={{cmb}} → '
          f'{len(sub)} cells.')

# ---------------------------------------------------------------------------
# Cell key canonicality — round-trip through CSV
# ---------------------------------------------------------------------------
def _row_from_plan(cell: dict) -> dict:
    """Synthetic CSV row matching what RunsCSVLogger would write for this cell."""
    key = run_grid.cell_key_from_plan(cell)
    (_, ce, act, pe, seed, ce_json) = key
    return {
        'dataset': cell['dataset'],
        'ce':      ce,
        'act':     act,
        'pe':      pe,
        'seed':    str(seed),
        'encoding_kwargs_json': ce_json,
        'status':  'completed',
    }


def test_cell_key_round_trip_main_grid() -> None:
    """A plan's cell-key matches the CSV-row's cell-key for every main cell."""
    print('\n[run_grid] cell-key round-trip: main grid ...')
    for c in run_grid.build_main_grid(seed=42):
        row = _row_from_plan(c)
        assert run_grid.cell_key_from_plan(c) == run_grid.cell_key_from_row(row), c
    print(f'  OK {EXPECTED_MAIN} main-grid cells round-trip through CSV.')


def test_cell_key_round_trip_sh_post_sh_pre() -> None:
    """sh_post_saturation and sh_pre_saturation cells (which override --sh_lmax) also round-trip."""
    print('\n[run_grid] cell-key round-trip: sh_post_saturation + sh_pre_saturation ...')
    for c in run_grid.build_sh_post_saturation_grid(42) + run_grid.build_sh_pre_saturation_grid(42):
        row = _row_from_plan(c)
        plan_key = run_grid.cell_key_from_plan(c)
        row_key = run_grid.cell_key_from_row(row)
        assert plan_key == row_key, f'mismatch for {c}: plan={plan_key} row={row_key}'
    print(f'  OK {EXPECTED_POST_SATURATION} sh_post_saturation + {EXPECTED_PRE_SATURATION} sh_pre_saturation cells round-trip.')

def test_sh_post_distinct_from_main_grid_sh_cell() -> None:
    """A main-grid SH cell (L_max=32 default) is a DIFFERENT cell from the
    sh_post_saturation matched cell on the same dataset (L_max=13 or 31)."""
    print('\n[run_grid] sh_post_saturation cells distinct from main-grid SH cells ...')
    main_cells = run_grid.build_main_grid(seed=42)
    sh_post_cells = run_grid.build_sh_post_saturation_grid(seed=42)
    for act_key in ACTIVATIONS:
        cell_key = f'{act_key}__sh'
        main_sh = next(c for c in main_cells
                       if c['cell_key'] == cell_key and c['dataset'] == 'era5')
        sh_post_sh = next(c for c in sh_post_cells
                      if c['cell_key'] == cell_key and c['dataset'] == 'era5')
        assert (run_grid.cell_key_from_plan(main_sh)
                != run_grid.cell_key_from_plan(sh_post_sh)), (
            f'main {main_sh} and sh_post {sh_post_sh} collided on the same cell key'
        )
    print('  OK main-grid (L_max=32) and sh_post_saturation (matched L_max) are distinct cells.')


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------
def _write_runs_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_TEST_CSV_COLUMNS,
                                extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, '') for col in _TEST_CSV_COLUMNS})


def test_resume_skips_completed_cells() -> None:
    """If some cells already have status='completed' in runs.csv, the launcher
    filters them out of `pending`. Failed-status rows are NOT filtered."""
    print('\n[run_grid] resume detection: completed cells filtered ...')
    main_cells = run_grid.build_main_grid(seed=42)
    first_complete = main_cells[0]
    second_failed = main_cells[1]

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'results', 'runs.csv')
        _write_runs_csv(csv_path, [
            {**_row_from_plan(first_complete),  'status': 'completed'},
            {**_row_from_plan(second_failed),   'status': 'oom'},
        ])
        completed = run_grid.load_completed_cells(csv_path)
        assert run_grid.cell_key_from_plan(first_complete) in completed
        assert run_grid.cell_key_from_plan(second_failed) not in completed, (
            'failed cells must NOT be treated as completed'
        )
        pending = [c for c in main_cells
                   if run_grid.cell_key_from_plan(c) not in completed]
        assert first_complete not in pending
        assert second_failed in pending
        print(f"  OK 1 completed + 1 failed: pending={len(pending)} of "
              f"{len(main_cells)}.")


def test_resume_empty_csv_runs_everything() -> None:
    """No runs.csv → every cell is pending."""
    print('\n[run_grid] resume detection: no runs.csv → all pending ...')
    completed = run_grid.load_completed_cells('/no/such/file.csv')
    assert completed == set()
    main_cells = run_grid.build_main_grid(seed=42)
    pending = [c for c in main_cells
               if run_grid.cell_key_from_plan(c) not in completed]
    assert len(pending) == len(main_cells)
    print(f'  OK no CSV → {len(pending)} pending (full grid).')

def test_resume_distinguishes_lmax_variants() -> None:
    """Marking the main-grid SH cell (L_max=32) as completed does NOT cause
    the sh_post_saturation SH cell (L_max=13) on the same dataset to be skipped."""
    print('\n[run_grid] resume detection: L_max variants distinct ...')
    main = run_grid.build_main_grid(seed=42)
    sh_post = run_grid.build_sh_post_saturation_grid(seed=42)

    main_sh_era5 = next(c for c in main
                        if c['cell_key'] == 'scaled_sine__sh'
                        and c['dataset'] == 'era5')
    sh_post_sh_era5 = next(c for c in sh_post
                       if c['cell_key'] == 'scaled_sine__sh'
                       and c['dataset'] == 'era5')

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        _write_runs_csv(csv_path, [_row_from_plan(main_sh_era5)])
        completed = run_grid.load_completed_cells(csv_path)
        assert run_grid.cell_key_from_plan(main_sh_era5) in completed
        assert run_grid.cell_key_from_plan(sh_post_sh_era5) not in completed, (
            'sh_post_saturation (L_max=13) should not be skipped when main-grid (L_max=32) '
            'is completed'
        )
        print('  OK main-grid (L_max=32) completed does not skip sh_post_saturation (L_max=13).')



# ---------------------------------------------------------------------------
# Save dir + describe helpers
# ---------------------------------------------------------------------------
def test_save_dir_includes_sub_grid_tag() -> None:
    """The per-cell save dir for sh_post_saturation/sh_pre_saturation includes the tag so they don't
    collide with the main-grid SH cell on the same (cell_key, dataset)."""
    print('\n[run_grid] cell_save_dir distinguishes main / sh_post_saturation / sh_pre_saturation ...')
    main = next(c for c in run_grid.build_main_grid(42)
                if c['cell_key'] == 'scaled_sine__sh'
                and c['dataset'] == 'era5')
    sh_post = next(c for c in run_grid.build_sh_post_saturation_grid(42)
               if c['cell_key'] == 'scaled_sine__sh'
               and c['dataset'] == 'era5')

    d_main = run_grid.cell_save_dir('logs/grid', main)
    d_sh_post = run_grid.cell_save_dir('logs/grid', sh_post)
    assert d_main != d_sh_post, f'collision: both went to {d_main}'
    assert 'sh_post' in d_sh_post, f'sh_post_saturation tag missing from {d_sh_post}'
    print(f'  OK main → {d_main}')
    print(f'     sh_post_saturation  → {d_sh_post}')



def test_describe_cell_contains_cell_key_and_dataset() -> None:
    print('\n[run_grid] describe_cell is human-readable ...')
    c = run_grid.build_main_grid(42)[0]
    s = run_grid.describe_cell(c)
    assert c['cell_key'] in s
    assert c['dataset'] in s
    print(f"  OK '{s}'")


def main() -> None:
    print('== Grid construction ==')
    test_main_grid_size_and_uniqueness()
    test_sh_post_saturation_grid()
    test_sh_pre_saturation_grid()
    test_full_grid_composes_subsets()
    test_filters_narrow_the_grid()

    print('\n== Cell-key canonicality ==')
    test_cell_key_round_trip_main_grid()
    test_cell_key_round_trip_sh_post_sh_pre()
    test_sh_post_distinct_from_main_grid_sh_cell()

    print('\n== Resume detection ==')
    test_resume_skips_completed_cells()
    test_resume_empty_csv_runs_everything()
    test_resume_distinguishes_lmax_variants()

    print('\n== Save dir + helpers ==')
    test_save_dir_includes_sub_grid_tag()
    test_describe_cell_contains_cell_key_and_dataset()

    print('\nAll run_grid tests passed.')


if __name__ == '__main__':
    main()

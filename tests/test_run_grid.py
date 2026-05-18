"""
Unit tests for scripts/run_grid.py — grid construction, cell-key canonicality,
and resume-detection against a synthetic runs.csv.

We do NOT spawn any subprocesses; all assertions are against the grid-building
and key-derivation logic.
"""
from __future__ import annotations

import csv
import json
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

from config.architectures import ARCHITECTURES                            # noqa: E402
from config.constants import CE_CHOICES, DATASET_CHOICES                  # noqa: E402

# Minimal column subset for synthetic CSVs: only the columns
# `cell_key_from_row` reads, plus `status` for resume filtering.
# (The real `RUNS_CSV_SCHEMA` in callbacks/runs_csv.py is much larger but
# requires torch via PL's Callback base class — keeping the test torch-free.)
_TEST_CSV_COLUMNS = (
    'dataset', 'ce', 'arch', 'act', 'mlp_act', 'kan_act',
    'pe', 'seed', 'encoding_kwargs_json', 'status',
)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def test_main_grid_size_and_uniqueness() -> None:
    """Main grid is exactly len(ARCH) × len(CE) × len(DATASETS) cells, unique."""
    print('\n[run_grid] main grid: shape & uniqueness ...')
    cells = run_grid.build_main_grid(seed=42)
    expected = len(ARCHITECTURES) * len(CE_CHOICES) * len(DATASET_CHOICES)
    assert len(cells) == expected, f'expected {expected} cells, got {len(cells)}'
    keys = {run_grid.cell_key_from_plan(c) for c in cells}
    assert len(keys) == len(cells), 'duplicate cell keys in main grid'
    print(f'  OK {expected} cells, all unique (4 archs × {len(CE_CHOICES)} ce × {len(DATASET_CHOICES)} ds).')


def test_h5a_grid_size_and_lmax_values() -> None:
    """H5a is 4 archs × 2 datasets × 1 L_max each = 8 cells."""
    print('\n[run_grid] H5a grid: 8 cells with matched L_max ...')
    cells = run_grid.build_h5a_grid(seed=42)
    assert len(cells) == 8, f'expected 8, got {len(cells)}'
    # All H5a cells are SH-encoded.
    assert all(c['ce'] == 'spherical-harmonics' for c in cells)
    # The L_max values for the two datasets:
    by_dataset = {}
    for c in cells:
        lmax_idx = c['extra_cli'].index('--sh_lmax') + 1
        by_dataset.setdefault(c['dataset'], set()).add(int(c['extra_cli'][lmax_idx]))
    assert by_dataset['era5'] == {13}, f'ERA5 L_max: {by_dataset["era5"]}'
    assert by_dataset['hdri_sky'] == {31}, f'HDRI_sky L_max: {by_dataset["hdri_sky"]}'
    print(f'  OK H5a: ERA5 L_max=13, HDRI_sky L_max=31, 4 archs each.')


def test_h5b_grid_size_and_lmax_values() -> None:
    """H5b is 4 archs × 3 high-bandwidth datasets × L_max=16 = 12 cells."""
    print('\n[run_grid] H5b grid: 12 cells at L_max=16 ...')
    cells = run_grid.build_h5b_grid(seed=42)
    assert len(cells) == 12, f'expected 12, got {len(cells)}'
    assert all(c['ce'] == 'spherical-harmonics' for c in cells)
    datasets = {c['dataset'] for c in cells}
    assert datasets == {'etopo1', 'hdri_urban', 'cmb'}, datasets
    for c in cells:
        lmax_idx = c['extra_cli'].index('--sh_lmax') + 1
        assert int(c['extra_cli'][lmax_idx]) == 16
    print(f'  OK H5b: L_max=16 for ETOPO1, HDRI_urban, CMB, 4 archs each.')


def test_total_grid_is_100() -> None:
    """The full grid is 80 + 8 + 12 = 100 cells (preregistration §3.5)."""
    print('\n[run_grid] grand total ...')
    n = (len(run_grid.build_main_grid(42))
         + len(run_grid.build_h5a_grid(42))
         + len(run_grid.build_h5b_grid(42)))
    assert n == 100, f'expected 100, got {n}'
    print(f'  OK 80 (main) + 8 (H5a) + 12 (H5b) = {n} runs.')


# ---------------------------------------------------------------------------
# Cell key canonicality — round-trip through CSV
# ---------------------------------------------------------------------------
def _row_from_plan(cell: dict) -> dict:
    """Synthetic CSV row matching what RunsCSVLogger would write for this cell."""
    arch_cfg = ARCHITECTURES[cell['arch_key']]
    # encoding_kwargs computed exactly the way the launcher's
    # cell_key_from_plan does (so the round-trip is meaningful).
    key = run_grid.cell_key_from_plan(cell)
    (_, ce, arch, act, mlp_act, kan_act, pe, seed, ce_json) = key
    return {
        'dataset': cell['dataset'],
        'ce':      ce,
        'arch':    arch,
        'act':     act,
        'mlp_act': mlp_act,
        'kan_act': kan_act,
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
    print('  OK 80 main-grid cells round-trip through CSV.')


def test_cell_key_round_trip_h5a_h5b() -> None:
    """H5a and H5b cells (which override --sh_lmax) also round-trip."""
    print('\n[run_grid] cell-key round-trip: H5a + H5b ...')
    for c in run_grid.build_h5a_grid(42) + run_grid.build_h5b_grid(42):
        row = _row_from_plan(c)
        plan_key = run_grid.cell_key_from_plan(c)
        row_key = run_grid.cell_key_from_row(row)
        assert plan_key == row_key, f'mismatch for {c}: plan={plan_key} row={row_key}'
    print('  OK 8 H5a + 12 H5b cells round-trip through CSV.')


def test_h5a_distinct_from_main_grid_sh_cell() -> None:
    """A main-grid SH cell (L_max=32 default) is a DIFFERENT cell from the
    H5a matched cell on the same dataset (L_max=13 or 31). They must have
    distinct cell keys, otherwise resume-detection would skip the H5a run."""
    print('\n[run_grid] H5a cells distinct from main-grid SH cells ...')
    main_cells = run_grid.build_main_grid(seed=42)
    h5a_cells = run_grid.build_h5a_grid(seed=42)
    # Pick the main-grid SH/ERA5 cell and the H5a SH/ERA5 cell — they share
    # dataset, encoding, architecture, but differ in encoding_kwargs.
    for arch_key in ARCHITECTURES:
        main_sh = next(c for c in main_cells
                       if c['arch_key'] == arch_key
                       and c['ce'] == 'spherical-harmonics'
                       and c['dataset'] == 'era5')
        h5a_sh = next(c for c in h5a_cells
                      if c['arch_key'] == arch_key and c['dataset'] == 'era5')
        assert (run_grid.cell_key_from_plan(main_sh)
                != run_grid.cell_key_from_plan(h5a_sh)), (
            f'main {main_sh} and h5a {h5a_sh} collided on the same cell key'
        )
    print('  OK main-grid (L_max=32) and H5a (L_max=13) are distinct cells.')


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
        # Filter behaviour: only first_complete should be skipped.
        pending = [c for c in main_cells
                   if run_grid.cell_key_from_plan(c) not in completed]
        assert first_complete not in pending
        assert second_failed in pending
        print(f"  OK 1 completed + 1 failed: pending={len(pending)} of "
              f"{len(main_cells)}.")


def test_resume_empty_csv_runs_everything() -> None:
    """No runs.csv (or empty completed set) → every cell is pending."""
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
    the H5a SH cell (L_max=13) on the same dataset to be skipped."""
    print('\n[run_grid] resume detection: L_max variants distinct ...')
    main = run_grid.build_main_grid(seed=42)
    h5a = run_grid.build_h5a_grid(seed=42)

    # Pick the main SH/ERA5 cell and the matching H5a SH/ERA5 cell.
    main_sh_era5 = next(c for c in main
                        if c['ce'] == 'spherical-harmonics'
                        and c['dataset'] == 'era5'
                        and c['arch_key'] == 'scaled_sine_mlp')
    h5a_sh_era5 = next(c for c in h5a
                       if c['dataset'] == 'era5'
                       and c['arch_key'] == 'scaled_sine_mlp')

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        _write_runs_csv(csv_path, [_row_from_plan(main_sh_era5)])
        completed = run_grid.load_completed_cells(csv_path)
        assert run_grid.cell_key_from_plan(main_sh_era5) in completed
        assert run_grid.cell_key_from_plan(h5a_sh_era5) not in completed, (
            'H5a (L_max=13) should not be skipped when main-grid (L_max=32) '
            'is completed'
        )
        print('  OK main-grid (L_max=32) completed does not skip H5a (L_max=13).')


# ---------------------------------------------------------------------------
# Save dir + describe helpers
# ---------------------------------------------------------------------------
def test_save_dir_includes_tag_for_h5() -> None:
    """The per-cell save dir for H5a/H5b includes the tag so they don't
    collide with the main-grid SH cell on the same (arch, dataset)."""
    print('\n[run_grid] cell_save_dir distinguishes main / H5a / H5b ...')
    main = next(c for c in run_grid.build_main_grid(42)
                if c['ce'] == 'spherical-harmonics'
                and c['dataset'] == 'era5'
                and c['arch_key'] == 'scaled_sine_mlp')
    h5a = next(c for c in run_grid.build_h5a_grid(42)
               if c['dataset'] == 'era5'
               and c['arch_key'] == 'scaled_sine_mlp')

    d_main = run_grid.cell_save_dir('logs/grid', main)
    d_h5a = run_grid.cell_save_dir('logs/grid', h5a)
    assert d_main != d_h5a, f'collision: both went to {d_main}'
    assert 'h5a' in d_h5a, f'H5a tag missing from {d_h5a}'
    print(f'  OK main → {d_main}')
    print(f'     H5a  → {d_h5a}')


def test_describe_cell_contains_arch_and_dataset() -> None:
    print('\n[run_grid] describe_cell is human-readable ...')
    c = run_grid.build_main_grid(42)[0]
    s = run_grid.describe_cell(c)
    assert c['arch_key'] in s
    assert c['ce'] in s
    assert c['dataset'] in s
    print(f"  OK '{s}'")


def main() -> None:
    print('== Grid construction ==')
    test_main_grid_size_and_uniqueness()
    test_h5a_grid_size_and_lmax_values()
    test_h5b_grid_size_and_lmax_values()
    test_total_grid_is_100()

    print('\n== Cell-key canonicality ==')
    test_cell_key_round_trip_main_grid()
    test_cell_key_round_trip_h5a_h5b()
    test_h5a_distinct_from_main_grid_sh_cell()

    print('\n== Resume detection ==')
    test_resume_skips_completed_cells()
    test_resume_empty_csv_runs_everything()
    test_resume_distinguishes_lmax_variants()

    print('\n== Save dir + helpers ==')
    test_save_dir_includes_tag_for_h5()
    test_describe_cell_contains_arch_and_dataset()

    print('\nAll run_grid tests passed.')


if __name__ == '__main__':
    main()

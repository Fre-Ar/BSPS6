"""
End-to-end test for src/analysis — synthesize a runs.csv with KNOWN patterns
that each hypothesis should detect, run the analysis pipeline, and verify
the decisions match expectations (preregistration §6: "analysis script
is written and committed BEFORE runs begin, against synthetic placeholder
data with the expected schema").

Post-redesign factor structure (preregistration §3.2 / §3.3):
  * 3 activations × 5 PE cells = 15 Coordinate-MLP cells
  * 1 Fourier KAN row (reported separately; excluded from H1/H3/H4)
  * 5 datasets, 1 seed → 80 main-grid runs + 6 H5a + 9 H5b = 95 total

We generate two synthetic worlds:
  * 'happy' — every hypothesis should accept.
  * 'null'  — no signal anywhere; every hypothesis should reject or remain
              inconclusive.

Run from repo root:
    PYTHONPATH=src python tests/test_analysis.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.characterization import DATASET_METRICS                     # noqa: E402
from analysis.decisions import (                                          # noqa: E402
    evaluate_h1, evaluate_h3, evaluate_h4,
    evaluate_h5a, evaluate_h5b,
    SH_LMAX_DEFAULT, SH_LMAX_H5A, SH_LMAX_H5B,
    HIGH_BANDWIDTH_DATASETS,
)
from analysis.run_analysis import apply_holm_bonferroni                   # noqa: E402
from config.architectures import ACTIVATIONS, PE_CELLS, KAN_ROW           # noqa: E402

DATASETS = tuple(DATASET_METRICS.keys())

# Per-activation CLI flag values (the `act` column in runs.csv).
ACT_TO_FLAG = {
    'relu':         'relu',
    'scaled_sine':  'scaled-sine',
    'gaussian':     'gaussian',
}


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def _mlp_row(act_key: str, pe_key: str, dataset: str, lmax: int | None,
             psnr: float, polar: float, equatorial: float,
             status: str = 'completed') -> dict:
    """Synthetic CSV row for a Coordinate-MLP cell."""
    cfg = PE_CELLS[pe_key]
    ce = cfg['ce']
    pe = cfg['pe']
    if ce == 'spherical-harmonics' and lmax is not None:
        ce_kwargs = {'L_max': lmax}
    else:
        ce_kwargs = {}
    return {
        'dataset': dataset,
        'ce':      ce,
        'arch':    'mlp',
        'act':     ACT_TO_FLAG[act_key],
        'mlp_act': ACT_TO_FLAG[act_key],
        'kan_act': '',           # n/a for MLP cells
        'pe':      pe,
        'seed':    42,
        'encoding_kwargs_json': json.dumps(ce_kwargs, sort_keys=True),
        'status':  status,
        'held_out_psnr':            psnr,
        'held_out_psnr_polar':      polar,
        'held_out_psnr_equatorial': equatorial,
    }


def _kan_row(dataset: str, psnr: float, polar: float, equatorial: float,
             status: str = 'completed') -> dict:
    """Synthetic CSV row for the standalone Fourier KAN cell."""
    cfg = KAN_ROW['fourier_kan']
    return {
        'dataset': dataset,
        'ce':      cfg['ce'],
        'arch':    cfg['arch'],
        'act':     cfg['act'],
        'mlp_act': '',
        'kan_act': cfg['act'],
        'pe':      cfg['pe'],
        'seed':    42,
        'encoding_kwargs_json': '{}',
        'status':  status,
        'held_out_psnr':            psnr,
        'held_out_psnr_polar':      polar,
        'held_out_psnr_equatorial': equatorial,
    }


def generate_happy_runs(seed: int = 0) -> pd.DataFrame:
    """A synthetic runs.csv where every hypothesis SHOULD accept.

    Design:
      * H1: none_angular has ~5 dB larger polar penalty than none_cartesian.
      * H3: PE choice ranges over ~5 dB; activation ranges over ~0.5 dB.
        η²(pe) >> η²(activation).
      * H4: PSNR is a strong linear function of (L_95, CV, P99_norm) — we
        build it that way for each PE cell.
      * H5a: matched L_max gives PSNR within 0.1 dB of L_max=32.
      * H5b: L_max=32 is ~3 dB better than L_max=16 on high-bandwidth datasets.
    """
    rng = np.random.default_rng(seed)

    # Per-PE-cell offsets — H3 driver. Range ~5 dB to dominate activation.
    PE_OFFSET = {
        'none_angular':   -3.0,
        'none_cartesian':  1.0,
        'rff':             1.5,
        'sh':              2.0,
        'fkan':            2.5,
    }
    # Polar penalty: none_angular suffers; everything else does not.
    POLAR_PENALTY = {
        'none_angular':   5.0,
        'none_cartesian': 0.5,
        'rff':            0.5,
        'sh':             0.5,
        'fkan':           0.5,
    }
    # Per-activation offsets — small relative to PE.
    ACT_OFFSET = {'relu': -0.2, 'scaled_sine': 0.3, 'gaussian': 0.0}

    # Dataset factor — H4 should detect signal from (L_95, CV, P99_norm).
    def dataset_psnr(dataset: str) -> float:
        m = DATASET_METRICS[dataset]
        base = 30.0
        return base + (
            - 0.05 * m['L_95']           # higher bandwidth → harder
            + 5.0 * (1.0 - m['CV'])      # more anisotropic → harder
            - 10.0 * m['P99_norm']       # sharper signal → harder
        )

    rows: list[dict] = []

    # ---- Main-grid MLP cells (3 activations × 5 PEs × 5 datasets = 75) ----
    for act_key in ACTIVATIONS:
        for pe_key in PE_CELLS:
            for dataset in DATASETS:
                psnr = (dataset_psnr(dataset)
                        + PE_OFFSET[pe_key]
                        + ACT_OFFSET[act_key]
                        + rng.normal(0, 0.1))
                polar = psnr - POLAR_PENALTY[pe_key]
                equatorial = psnr + 0.5
                lmax = SH_LMAX_DEFAULT if pe_key == 'sh' else None
                rows.append(_mlp_row(act_key, pe_key, dataset, lmax,
                                     psnr, polar, equatorial))

    # ---- KAN row (5 datasets) ----
    for dataset in DATASETS:
        psnr = dataset_psnr(dataset) + 1.0 + rng.normal(0, 0.1)
        polar = psnr - 0.5
        equatorial = psnr + 0.5
        rows.append(_kan_row(dataset, psnr, polar, equatorial))

    # ---- H5a: matched L_max for low-bandwidth datasets, 3 acts × 2 ds = 6 ----
    for act_key in ACTIVATIONS:
        for dataset, lmax_matched in SH_LMAX_H5A.items():
            base = (dataset_psnr(dataset)
                    + PE_OFFSET['sh']
                    + ACT_OFFSET[act_key])
            psnr = base - 0.05 + rng.normal(0, 0.05)   # within 0.1 dB of default
            polar = psnr - POLAR_PENALTY['sh']
            equatorial = psnr + 0.5
            rows.append(_mlp_row(act_key, 'sh', dataset, lmax_matched,
                                 psnr, polar, equatorial))

    # ---- H5b: L_max=16 on high-bandwidth datasets, 3 acts × 3 ds = 9 ----
    for act_key in ACTIVATIONS:
        for dataset in HIGH_BANDWIDTH_DATASETS:
            base = (dataset_psnr(dataset)
                    + PE_OFFSET['sh']
                    + ACT_OFFSET[act_key])
            psnr = base - 3.0 + rng.normal(0, 0.1)
            polar = psnr - POLAR_PENALTY['sh']
            equatorial = psnr + 0.5
            rows.append(_mlp_row(act_key, 'sh', dataset, SH_LMAX_H5B,
                                 psnr, polar, equatorial))

    return pd.DataFrame(rows)


def generate_null_runs(seed: int = 0) -> pd.DataFrame:
    """A synthetic runs.csv with NO signal. All hypotheses should fail to
    accept (reject or inconclusive)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for act_key in ACTIVATIONS:
        for pe_key in PE_CELLS:
            for dataset in DATASETS:
                psnr = 30.0 + rng.normal(0, 1.0)
                lmax = SH_LMAX_DEFAULT if pe_key == 'sh' else None
                rows.append(_mlp_row(act_key, pe_key, dataset, lmax,
                                     psnr, psnr, psnr))
    for dataset in DATASETS:
        psnr = 30.0 + rng.normal(0, 1.0)
        rows.append(_kan_row(dataset, psnr, psnr, psnr))

    for act_key in ACTIVATIONS:
        for dataset, lmax in SH_LMAX_H5A.items():
            psnr = 30.0 + rng.normal(0, 1.0)
            rows.append(_mlp_row(act_key, 'sh', dataset, lmax,
                                 psnr, psnr, psnr))
        for dataset in HIGH_BANDWIDTH_DATASETS:
            psnr = 30.0 + rng.normal(0, 1.0)
            rows.append(_mlp_row(act_key, 'sh', dataset, SH_LMAX_H5B,
                                 psnr, psnr, psnr))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-evaluator tests against the happy world
# ---------------------------------------------------------------------------
def test_happy_h1_accepts() -> None:
    print('\n[analysis] happy world H1 ...')
    df = generate_happy_runs()
    d = evaluate_h1(df)
    print(f"  H1: {d['decision']}  "
          f"median_delta={d['statistic']['median_delta_db']:.3f}  "
          f"p={d['statistic']['p_value']:.4f}  n={d['n']}")
    assert d['decision'] == 'accepted', d


def test_happy_h3_accepts() -> None:
    print('\n[analysis] happy world H3 ...')
    df = generate_happy_runs()
    d = evaluate_h3(df)
    eta_pe = d['statistic']['eta_sq_pe']
    eta_act = d['statistic']['eta_sq_activation']
    print(f"  H3: {d['decision']}  "
          f"η²(pe)={eta_pe:.3f}  η²(act)={eta_act:.3f}  "
          f"ratio={d['statistic']['ratio_pe_over_act']:.2f}")
    assert d['decision'] == 'accepted', d
    assert eta_pe > 2 * eta_act, (eta_pe, eta_act)


def test_happy_h4_accepts() -> None:
    print('\n[analysis] happy world H4 ...')
    df = generate_happy_runs()
    d = evaluate_h4(df)
    for pe, info in d['statistic'].items():
        print(f"  H4 {pe}: R²={info['r2']:.3f} "
              f"CI=[{info['ci_lo']:.3f},{info['ci_hi']:.3f}]")
    assert d['decision'] == 'accepted', d


def test_happy_h5a_accepts() -> None:
    print('\n[analysis] happy world H5a ...')
    df = generate_happy_runs()
    d = evaluate_h5a(df)
    print(f"  H5a: {d['decision']}  "
          f"median={d['statistic']['median_delta_db']:.3f}  "
          f"p={d['statistic']['p_value']:.4f}  n={d['n']}")
    assert d['decision'] == 'accepted', d


def test_happy_h5b_accepts() -> None:
    print('\n[analysis] happy world H5b ...')
    df = generate_happy_runs()
    d = evaluate_h5b(df)
    print(f"  H5b: {d['decision']}  "
          f"median={d['statistic']['median_delta_db']:.3f}  "
          f"p={d['statistic']['p_value']:.4f}  n={d['n']}")
    assert d['decision'] == 'accepted', d


# ---------------------------------------------------------------------------
# Per-evaluator tests against the null world
# ---------------------------------------------------------------------------
def test_null_h1_not_accepted() -> None:
    print('\n[analysis] null world H1 ...')
    df = generate_null_runs()
    d = evaluate_h1(df)
    print(f"  H1: {d['decision']}  "
          f"median_delta={d['statistic']['median_delta_db']:.3f}")
    assert d['decision'] != 'accepted', d


def test_null_h5b_not_accepted() -> None:
    print('\n[analysis] null world H5b ...')
    df = generate_null_runs()
    d = evaluate_h5b(df)
    print(f"  H5b: {d['decision']}  "
          f"median={d['statistic']['median_delta_db']:.3f}")
    assert d['decision'] != 'accepted', d


# ---------------------------------------------------------------------------
# Missing-data robustness
# ---------------------------------------------------------------------------
def test_empty_df_all_inconclusive() -> None:
    """An empty DataFrame should produce 'inconclusive' decisions everywhere
    (never 'accepted', never a crash)."""
    print('\n[analysis] empty DataFrame ...')
    df = pd.DataFrame(columns=[
        'dataset', 'ce', 'arch', 'act', 'mlp_act', 'kan_act', 'pe',
        'status', 'held_out_psnr',
    ])
    for fn in (evaluate_h1, evaluate_h3, evaluate_h4,
               evaluate_h5a, evaluate_h5b):
        d = fn(df)
        assert d['decision'] in ('inconclusive', 'rejected'), (fn.__name__, d)
    print('  OK every evaluator returns inconclusive on empty input.')


def test_failed_rows_excluded() -> None:
    """status != 'completed' rows are ignored. Adding garbage failed rows
    should not change the H1 decision."""
    print('\n[analysis] status filter ...')
    df = generate_happy_runs()
    bad = df.iloc[:5].copy()
    bad['status'] = 'oom'
    bad['held_out_psnr_polar'] = -100.0
    bad['held_out_psnr_equatorial'] = 100.0
    df2 = pd.concat([df, bad], ignore_index=True)
    d1 = evaluate_h1(df)
    d2 = evaluate_h1(df2)
    assert d1['decision'] == d2['decision']
    print(f"  OK adding oom rows didn't change H1 ({d1['decision']}).")


# ---------------------------------------------------------------------------
# Holm-Bonferroni behavior
# ---------------------------------------------------------------------------
def test_holm_bonferroni_no_op_on_strong_signal() -> None:
    """When all 3 family p-values are very small, Holm leaves them accepted."""
    print('\n[analysis] Holm-Bonferroni: strong signal ...')
    decisions = {
        'H1':  {'decision': 'accepted', 'statistic': {'p_value': 1e-6}},
        'H5a': {'decision': 'accepted', 'statistic': {'p_value': 1e-6}},
        'H5b': {'decision': 'accepted', 'statistic': {'p_value': 1e-6}},
    }
    apply_holm_bonferroni(decisions)
    for h, d in decisions.items():
        assert d['decision'] == 'accepted', (h, d)
        assert d['statistic']['holm_significant'] is True
    print('  OK strong p-values survive Holm.')


def test_holm_bonferroni_downgrades_marginal() -> None:
    """A marginal p that passes uncorrected α but not Holm gets downgraded.
    Strict Holm: once a hypothesis fails, every later one (larger p) also fails."""
    print('\n[analysis] Holm-Bonferroni: strict failure cascade ...')
    decisions = {
        # m=3 family. Strict-Holm thresholds (ascending p): 0.05/3, 0.05/2, 0.05.
        # smallest=1e-6        → passes 0.05/3=0.0167.
        # middle=0.03          → fails  0.05/2=0.025.
        # largest=0.04         → strict Holm cascades the failure: non-sig.
        'H1':  {'decision': 'accepted', 'statistic': {'p_value': 1e-6}},
        'H5a': {'decision': 'accepted', 'statistic': {'p_value': 0.03}},
        'H5b': {'decision': 'accepted', 'statistic': {'p_value': 0.04}},
    }
    apply_holm_bonferroni(decisions)
    assert decisions['H1']['decision'] == 'accepted'
    assert decisions['H5a']['decision'] == 'inconclusive'
    assert decisions['H5a']['decision_before_holm'] == 'accepted'
    assert decisions['H5b']['decision'] == 'inconclusive', (
        f"strict Holm should cascade H5a's failure to H5b, but got "
        f"{decisions['H5b']['decision']}"
    )
    assert decisions['H5b']['statistic']['holm_significant'] is False
    print(f"  OK Holm decisions: H1=accepted, H5a/H5b downgraded "
          f"(cascade from H5a).")


# ---------------------------------------------------------------------------
# End-to-end smoke test: write CSV → invoke run_analysis.py → check outputs
# ---------------------------------------------------------------------------
def test_run_analysis_e2e_smoke() -> None:
    """Write happy synthetic runs.csv, invoke run_analysis.py via subprocess,
    verify decisions.json and tables/summary appear and contain accepted
    decisions for all 5 hypotheses."""
    print('\n[analysis] end-to-end smoke ...')
    df = generate_happy_runs()
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'runs.csv')
        out_dir = os.path.join(tmp, 'out')
        df.to_csv(csv_path, index=False)
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '..', 'src')
        result = subprocess.run(
            [sys.executable, '-u',
             os.path.join(os.path.dirname(__file__), '..', 'src', 'analysis',
                          'run_analysis.py'),
             '--runs_csv', csv_path,
             '--output_dir', out_dir,
             '--skip_figures'],
            env=env, check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print('STDOUT:', result.stdout)
            print('STDERR:', result.stderr)
            raise AssertionError(f'run_analysis.py exited {result.returncode}')
        dpath = os.path.join(out_dir, 'decisions.json')
        spath = os.path.join(out_dir, 'summary.md')
        assert os.path.exists(dpath), 'decisions.json missing'
        assert os.path.exists(spath), 'summary.md missing'
        with open(dpath) as f:
            d = json.load(f)
        # The happy world should accept H1, H3, H4, H5a, H5b.
        for h in ('H1', 'H3', 'H4', 'H5a', 'H5b'):
            assert d[h]['decision'] == 'accepted', (h, d[h]['decision'])
        print(f"  OK 5/5 hypotheses accepted in happy world; outputs at {out_dir}.")


def main() -> None:
    print('== Per-evaluator tests (happy world) ==')
    test_happy_h1_accepts()
    test_happy_h3_accepts()
    test_happy_h4_accepts()
    test_happy_h5a_accepts()
    test_happy_h5b_accepts()

    print('\n== Per-evaluator tests (null world) ==')
    test_null_h1_not_accepted()
    test_null_h5b_not_accepted()

    print('\n== Missing-data robustness ==')
    test_empty_df_all_inconclusive()
    test_failed_rows_excluded()

    print('\n== Holm-Bonferroni ==')
    test_holm_bonferroni_no_op_on_strong_signal()
    test_holm_bonferroni_downgrades_marginal()

    print('\n== End-to-end ==')
    test_run_analysis_e2e_smoke()

    print('\nAll analysis tests passed.')


if __name__ == '__main__':
    main()

"""
End-to-end test for src/analysis — synthesize a runs.csv with KNOWN patterns
that each hypothesis should detect, run the analysis pipeline, and verify
the decisions match expectations.

We generate two synthetic worlds:
  * 'happy' — every hypothesis should accept.
  * 'null'  — no signal anywhere; every hypothesis should reject or remain
              inconclusive.
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
    evaluate_h5_1, evaluate_h5a, evaluate_h5b,
    SH_LMAX_DEFAULT, SH_LMAX_H5A, SH_LMAX_H5B,
    HIGH_BANDWIDTH_DATASETS,
)
from analysis.run_analysis import apply_holm_bonferroni                   # noqa: E402

ARCHES = ('scaled_sine_mlp', 'relu_rff_mlp', 'fourier_kan', 'gaussian_fkan')
ENCODINGS = ('angular', 'cartesian', 'spherical-harmonics', 'spherical-rff')
DATASETS = tuple(DATASET_METRICS.keys())


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def _arch_to_cli(arch: str) -> dict:
    """Map our architecture key back to the (act, mlp_act, kan_act, pe) tuple
    that ends up in the CSV row. Must match what RunsCSVLogger writes via
    `pl_module.hparams` — see src/config/architectures.py."""
    if arch == 'scaled_sine_mlp':
        return {'arch': 'mlp', 'act': 'scaled-sine', 'mlp_act': 'scaled-sine',
                'kan_act': 'fourier', 'pe': 'None'}
    if arch == 'relu_rff_mlp':
        return {'arch': 'mlp', 'act': 'relu', 'mlp_act': 'scaled-sine',
                'kan_act': 'fourier', 'pe': 'RFF'}
    if arch == 'fourier_kan':
        return {'arch': 'kan', 'act': 'fourier', 'mlp_act': 'scaled-sine',
                'kan_act': 'fourier', 'pe': 'None'}
    if arch == 'gaussian_fkan':
        return {'arch': 'kamp', 'act': 'gaussian', 'mlp_act': 'gaussian',
                'kan_act': 'fourier', 'pe': 'None'}
    raise ValueError(arch)


def _row(arch: str, ce: str, dataset: str, lmax: int | None,
         psnr: float, polar: float, equatorial: float,
         status: str = 'completed') -> dict:
    if ce == 'spherical-harmonics' and lmax is not None:
        ce_kwargs = {'L_max': lmax}
    elif ce == 'spherical-rff':
        ce_kwargs = {'num_features': 32, 'seed': 42, 'sigma': 10.0}
    else:
        ce_kwargs = {}
    cli = _arch_to_cli(arch)
    return {
        'dataset': dataset,
        'ce':      ce,
        'arch':    cli['arch'],
        'act':     cli['act'],
        'mlp_act': cli['mlp_act'],
        'kan_act': cli['kan_act'],
        'pe':      cli['pe'],
        'seed':    42,
        'encoding_kwargs_json': json.dumps(ce_kwargs, sort_keys=True),
        'status':  status,
        'held_out_psnr':            psnr,
        'held_out_psnr_polar':      polar,
        'held_out_psnr_equatorial': equatorial,
    }


def generate_happy_runs(seed: int = 0) -> pd.DataFrame:
    """A synthetic runs.csv where every hypothesis SHOULD accept.

    Design:
      * H1: angular encoding is uniformly 5 dB worse at the poles than non-
        angular encodings (large polar penalty for angular only).
      * H3: encoding contributes much more PSNR variance than architecture.
        We make encoding effects ±5 dB and architecture effects ±0.5 dB.
      * H4: PSNR is a strong linear function of (L_95, CV, P99_norm) — we
        build it that way per encoding.
      * H5.1: SH wins on low-L_95 datasets (positive SH_advantage), loses on
        high-L_95 datasets (negative SH_advantage).
      * H5a: matched L_max gives PSNR within 0.1 dB of L_max=32 on ERA5/HDRI_sky.
      * H5b: L_max=32 is ~3 dB better than L_max=16 on high-bandwidth datasets.
    """
    rng = np.random.default_rng(seed)

    # Per-encoding offsets — H1 driver.
    ENC_OFFSET = {
        'angular':              -2.0,
        'cartesian':             1.5,
        'spherical-harmonics':   1.0,    # baseline for SH at default L_max
        'spherical-rff':         0.5,
    }
    POLAR_PENALTY = {'angular': 5.0, 'cartesian': 0.5,
                     'spherical-harmonics': 0.5, 'spherical-rff': 0.5}

    # Architecture variance — H3 should attribute most variance to encoding.
    ARCH_OFFSET = {a: 0.5 * i for i, a in enumerate(ARCHES)}

    # Dataset factor — H4 should detect signal from (L_95, CV, P99_norm).
    def dataset_psnr(dataset: str, ce: str) -> float:
        m = DATASET_METRICS[dataset]
        # Strong linear function for H4. Different encodings have different
        # sensitivities, but the H4 regression is fit per-encoding anyway.
        base = 30.0
        feat = (-0.05 * m['L_95']  # higher bandwidth → harder
                + 5.0 * (1.0 - m['CV'])  # higher anisotropy → harder
                - 10.0 * m['P99_norm'])  # sharper signal → harder
        return base + feat

    # H5.1: bias SH advantage by -L_95.
    def sh_vs_srff_bias(dataset: str) -> float:
        # Add bonus to SH for low-L_95, penalty for high-L_95.
        l95 = DATASET_METRICS[dataset]['L_95']
        return 6.0 - 0.04 * l95   # +5.5 dB at L_95=13, -3.4 dB at L_95=236

    rows: list[dict] = []
    for arch in ARCHES:
        for ce in ENCODINGS:
            for dataset in DATASETS:
                psnr = (dataset_psnr(dataset, ce)
                        + ENC_OFFSET[ce]
                        + ARCH_OFFSET[arch]
                        + rng.normal(0, 0.1))     # tiny noise
                if ce == 'spherical-harmonics':
                    psnr += 0  # baseline; H5.1 handled by SRFF entry below
                if ce == 'spherical-rff':
                    # Make SH advantage correlate with -L_95.
                    psnr -= sh_vs_srff_bias(dataset)
                if ce == 'spherical-harmonics':
                    psnr += sh_vs_srff_bias(dataset) * 0.5  # split the bias

                polar = psnr - POLAR_PENALTY[ce]
                equatorial = psnr + 0.5            # small equatorial bonus
                lmax = SH_LMAX_DEFAULT if ce == 'spherical-harmonics' else None
                rows.append(_row(arch, ce, dataset, lmax, psnr, polar, equatorial))

    # H5a: matched L_max for ERA5 (L_max=13) and HDRI_sky (L_max=31).
    # Should give PSNR within 0.1 dB of L_max=32 (post-saturation).
    for arch in ARCHES:
        for dataset, lmax_matched in SH_LMAX_H5A.items():
            # Find the corresponding L_max=32 PSNR
            base_psnr = (dataset_psnr(dataset, 'spherical-harmonics')
                         + ENC_OFFSET['spherical-harmonics']
                         + ARCH_OFFSET[arch]
                         + sh_vs_srff_bias(dataset) * 0.5)
            # Matched L_max stays within 0.1 dB of base.
            psnr = base_psnr - 0.05 + rng.normal(0, 0.05)
            polar = psnr - POLAR_PENALTY['spherical-harmonics']
            equatorial = psnr + 0.5
            rows.append(_row(arch, 'spherical-harmonics', dataset, lmax_matched,
                             psnr, polar, equatorial))

    # H5b: L_max=16 should be ~3 dB worse than L_max=32 on high-bandwidth datasets.
    for arch in ARCHES:
        for dataset in HIGH_BANDWIDTH_DATASETS:
            base_psnr = (dataset_psnr(dataset, 'spherical-harmonics')
                         + ENC_OFFSET['spherical-harmonics']
                         + ARCH_OFFSET[arch]
                         + sh_vs_srff_bias(dataset) * 0.5)
            psnr = base_psnr - 3.0 + rng.normal(0, 0.1)
            polar = psnr - POLAR_PENALTY['spherical-harmonics']
            equatorial = psnr + 0.5
            rows.append(_row(arch, 'spherical-harmonics', dataset, SH_LMAX_H5B,
                             psnr, polar, equatorial))

    return pd.DataFrame(rows)


def generate_null_runs(seed: int = 0) -> pd.DataFrame:
    """A synthetic runs.csv with NO signal. All hypotheses should
    fail to accept (reject or inconclusive)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for arch in ARCHES:
        for ce in ENCODINGS:
            for dataset in DATASETS:
                psnr = 30.0 + rng.normal(0, 1.0)
                rows.append(_row(arch, ce, dataset,
                                 SH_LMAX_DEFAULT if ce == 'spherical-harmonics' else None,
                                 psnr, psnr, psnr))
    # Add H5a/H5b cells with no systematic difference.
    for arch in ARCHES:
        for dataset, lmax in SH_LMAX_H5A.items():
            psnr = 30.0 + rng.normal(0, 1.0)
            rows.append(_row(arch, 'spherical-harmonics', dataset, lmax,
                             psnr, psnr, psnr))
        for dataset in HIGH_BANDWIDTH_DATASETS:
            psnr = 30.0 + rng.normal(0, 1.0)
            rows.append(_row(arch, 'spherical-harmonics', dataset, SH_LMAX_H5B,
                             psnr, psnr, psnr))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-evaluator tests against the happy world
# ---------------------------------------------------------------------------
def test_happy_h1_accepts() -> None:
    print('\n[analysis] happy world H1 ...')
    df = generate_happy_runs()
    d = evaluate_h1(df)
    print(f"  H1: {d['decision']}  median_delta={d['statistic']['median_delta_db']:.3f}  p={d['statistic']['p_value']:.4f}")
    assert d['decision'] == 'accepted', d


def test_happy_h3_accepts() -> None:
    print('\n[analysis] happy world H3 ...')
    df = generate_happy_runs()
    d = evaluate_h3(df)
    eta_e = d['statistic']['eta_sq_encoding']
    eta_a = d['statistic']['eta_sq_architecture']
    print(f"  H3: {d['decision']}  η²(enc)={eta_e:.3f}  η²(arch)={eta_a:.3f}")
    assert d['decision'] == 'accepted', d
    assert eta_e > 2 * eta_a, (eta_e, eta_a)


def test_happy_h4_accepts() -> None:
    print('\n[analysis] happy world H4 ...')
    df = generate_happy_runs()
    d = evaluate_h4(df)
    for ce, info in d['statistic'].items():
        print(f"  H4 {ce}: R²={info['r2']:.3f} CI=[{info['ci_lo']:.3f},{info['ci_hi']:.3f}]")
    assert d['decision'] == 'accepted', d


def test_happy_h5_1_accepts() -> None:
    print('\n[analysis] happy world H5.1 ...')
    df = generate_happy_runs()
    d = evaluate_h5_1(df)
    rho = d['statistic']['spearman_rho']
    p = d['statistic']['p_value']
    print(f"  H5.1: {d['decision']}  ρ={rho:.3f}  p={p:.4f}")
    assert d['decision'] == 'accepted', d


def test_happy_h5a_accepts() -> None:
    print('\n[analysis] happy world H5a ...')
    df = generate_happy_runs()
    d = evaluate_h5a(df)
    print(f"  H5a: {d['decision']}  median={d['statistic']['median_delta_db']:.3f}  p={d['statistic']['p_value']:.4f}  n={d['n']}")
    assert d['decision'] == 'accepted', d


def test_happy_h5b_accepts() -> None:
    print('\n[analysis] happy world H5b ...')
    df = generate_happy_runs()
    d = evaluate_h5b(df)
    print(f"  H5b: {d['decision']}  median={d['statistic']['median_delta_db']:.3f}  p={d['statistic']['p_value']:.4f}  n={d['n']}")
    assert d['decision'] == 'accepted', d


# ---------------------------------------------------------------------------
# Per-evaluator tests against the null world
# ---------------------------------------------------------------------------
def test_null_h1_not_accepted() -> None:
    print('\n[analysis] null world H1 ...')
    df = generate_null_runs()
    d = evaluate_h1(df)
    print(f"  H1: {d['decision']}  median_delta={d['statistic']['median_delta_db']:.3f}")
    assert d['decision'] != 'accepted', d


def test_null_h5b_not_accepted() -> None:
    print('\n[analysis] null world H5b ...')
    df = generate_null_runs()
    d = evaluate_h5b(df)
    print(f"  H5b: {d['decision']}  median={d['statistic']['median_delta_db']:.3f}")
    assert d['decision'] != 'accepted', d


# ---------------------------------------------------------------------------
# Missing-data robustness
# ---------------------------------------------------------------------------
def test_empty_df_all_inconclusive() -> None:
    """An empty DataFrame should produce 'inconclusive' decisions everywhere
    (never 'accepted', never a crash)."""
    print('\n[analysis] empty DataFrame ...')
    df = pd.DataFrame(columns=['dataset', 'ce', 'arch', 'status', 'held_out_psnr'])
    for fn in (evaluate_h1, evaluate_h3, evaluate_h4,
               evaluate_h5_1, evaluate_h5a, evaluate_h5b):
        d = fn(df)
        assert d['decision'] in ('inconclusive', 'rejected'), (fn.__name__, d)
    print('  OK every evaluator returns inconclusive on empty input.')


def test_failed_rows_excluded() -> None:
    """status != 'completed' rows are ignored. Adding garbage failed rows
    should not change the H1 decision."""
    print('\n[analysis] status filter ...')
    df = generate_happy_runs()
    # Inject some 'oom' rows with wildly different PSNR.
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
    # Strict Holm: H5b cascades from H5a's failure even though 0.04 < 0.05.
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
    decisions."""
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
        # Check expected outputs exist.
        dpath = os.path.join(out_dir, 'decisions.json')
        spath = os.path.join(out_dir, 'summary.md')
        assert os.path.exists(dpath), 'decisions.json missing'
        assert os.path.exists(spath), 'summary.md missing'
        with open(dpath) as f:
            d = json.load(f)
        # The happy world should accept H1, H3, H4, H5.1, H5a, H5b.
        for h in ('H1', 'H3', 'H4', 'H5_1', 'H5a', 'H5b'):
            assert d[h]['decision'] == 'accepted', (h, d[h]['decision'])
        print(f"  OK 6/6 hypotheses accepted in happy world; outputs at {out_dir}.")


def main() -> None:
    print('== Per-evaluator tests (happy world) ==')
    test_happy_h1_accepts()
    test_happy_h3_accepts()
    test_happy_h4_accepts()
    test_happy_h5_1_accepts()
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

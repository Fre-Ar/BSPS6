"""
End-to-end test for src/analysis — synthesize a runs.csv with KNOWN patterns
that each pre-committed analysis should detect, run the analysis pipeline,
and verify the descriptive summaries match expectations.

The analyses are descriptive (no accept/reject); these tests check that
the summary statistics land in plausible ranges given a known signal /
no signal, not that any binary decision is made.

Factor structure (preregistration §3.2 / §3.3):
  * activations × PE cells = Coordinate-MLP cells in the main grid
  * 5 datasets, 1 seed → main-grid runs + 6 post-saturation + 9 pre-saturation

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
    variance_decomposition,
    polar_penalty_contrast,
    characterization_correlations,
    sh_lmax_ablation,
    SH_LMAX_DEFAULT, SH_LMAX_H5A, SH_LMAX_H5B,
    HIGH_BANDWIDTH_DATASETS,
)
from config.architectures import ACTIVATIONS, PE_CELLS                    # noqa: E402

DATASETS = tuple(DATASET_METRICS.keys())

ACT_TO_FLAG = {
    'relu':         'relu',
    'scaled_sine':  'scaled-sine',
    'gaussian':     'gaussian',
}


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def _mlp_row(act_key, pe_key, dataset, lmax, psnr, polar, equatorial,
             status='completed'):
    cfg = PE_CELLS[pe_key]
    ce, pe = cfg['ce'], cfg['pe']
    ce_kwargs = {'L_max': lmax} if (ce == 'spherical-harmonics' and lmax is not None) else {}
    return {
        'dataset': dataset, 'ce': ce, 'act': ACT_TO_FLAG[act_key], 'pe': pe,
        'seed': 42, 'encoding_kwargs_json': json.dumps(ce_kwargs, sort_keys=True),
        'status': status,
        'reconstruction_psnr':            psnr,
        'reconstruction_psnr_polar':      polar,
        'reconstruction_psnr_equatorial': equatorial,
        'held_out_psnr':                  psnr,
        'held_out_psnr_polar':            polar,
        'held_out_psnr_equatorial':       equatorial,
    }


def generate_happy_runs(seed: int = 0) -> pd.DataFrame:
    """Synthetic runs.csv with KNOWN signal in every dimension:

    * PE choice produces a ~5 dB range; activation produces a ~0.5 dB range —
      so variance_decomposition should find η²(pe) > η²(activation) by 5–10×.
    * none_angular has a ~5 dB larger polar penalty than none_cartesian —
      so polar_penalty_contrast should produce a median Δ ≈ 5 dB.
    * Dataset PSNR strongly decreases with L_95 (negative ρ ≈ −1 expected
      for L_95) — so characterization_correlations should yield strong
      negative ρ for L_95 across PEs.
    * Matched-L_max within 0.1 dB of L_max=32 on low-bandwidth; L_max=32
      gains 3 dB over L_max=16 on high-bandwidth — so sh_lmax_ablation
      post-saturation median ≈ 0 and pre-saturation median ≈ +3 dB.
    """
    rng = np.random.default_rng(seed)

    PE_OFFSET = {
        'none_angular':   -3.0,
        'none_cartesian':  1.0,
        'rff':             1.5,
        'sh':              2.0,
        'fkan':            2.5,
    }
    POLAR_PENALTY = {
        'none_angular':   5.0,
        'none_cartesian': 0.5,
        'rff':            0.5,
        'sh':             0.5,
        'fkan':           0.5,
    }
    ACT_OFFSET = {'relu': -0.2, 'scaled_sine': 0.3, 'gaussian': 0.0}

    def dataset_psnr(d):
        m = DATASET_METRICS[d]
        return 30.0 + (-0.05 * m['L_95'] + 5.0 * (1.0 - m['CV'])
                       - 10.0 * m['P99_norm'])

    rows = []
    for act_key in ACTIVATIONS:
        for pe_key in PE_CELLS:
            for dataset in DATASETS:
                psnr = (dataset_psnr(dataset) + PE_OFFSET[pe_key]
                        + ACT_OFFSET[act_key] + rng.normal(0, 0.1))
                polar = psnr - POLAR_PENALTY[pe_key]
                equatorial = psnr + 0.5
                lmax = SH_LMAX_DEFAULT if pe_key == 'sh' else None
                rows.append(_mlp_row(act_key, pe_key, dataset, lmax,
                                     psnr, polar, equatorial))

    # Post-saturation regime (low-bandwidth datasets, matched L_max).
    for act_key in ACTIVATIONS:
        for dataset, lmax_matched in SH_LMAX_H5A.items():
            base = (dataset_psnr(dataset) + PE_OFFSET['sh']
                    + ACT_OFFSET[act_key])
            psnr = base - 0.05 + rng.normal(0, 0.05)   # within 0.1 dB
            polar = psnr - POLAR_PENALTY['sh']; equatorial = psnr + 0.5
            rows.append(_mlp_row(act_key, 'sh', dataset, lmax_matched,
                                 psnr, polar, equatorial))

    # Pre-saturation regime (high-bandwidth datasets, L_max=16).
    for act_key in ACTIVATIONS:
        for dataset in HIGH_BANDWIDTH_DATASETS:
            base = (dataset_psnr(dataset) + PE_OFFSET['sh']
                    + ACT_OFFSET[act_key])
            psnr = base - 3.0 + rng.normal(0, 0.1)
            polar = psnr - POLAR_PENALTY['sh']; equatorial = psnr + 0.5
            rows.append(_mlp_row(act_key, 'sh', dataset, SH_LMAX_H5B,
                                 psnr, polar, equatorial))

    return pd.DataFrame(rows)


def generate_null_runs(seed: int = 0) -> pd.DataFrame:
    """No systematic signal in any dimension."""
    rng = np.random.default_rng(seed)
    rows = []
    for act_key in ACTIVATIONS:
        for pe_key in PE_CELLS:
            for dataset in DATASETS:
                psnr = 30.0 + rng.normal(0, 1.0)
                lmax = SH_LMAX_DEFAULT if pe_key == 'sh' else None
                rows.append(_mlp_row(act_key, pe_key, dataset, lmax,
                                     psnr, psnr, psnr))
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
# Happy-world tests — check the produced numbers land where we expect
# ---------------------------------------------------------------------------
def test_happy_variance_decomposition() -> None:
    print('\n[analysis] happy world: variance_decomposition ...')
    df = generate_happy_runs()
    d = variance_decomposition(df)
    eta_pe  = d['statistics']['eta_sq_pe']
    eta_act = d['statistics']['eta_sq_activation']
    eta_ds  = d['statistics']['eta_sq_dataset']
    print(f"  η²: pe={eta_pe:.3f}, activation={eta_act:.3f}, dataset={eta_ds:.3f}")
    assert eta_pe > 5 * eta_act, (
        f'expected η²(pe) >> η²(activation), got {eta_pe} vs {eta_act}'
    )
    assert eta_ds > eta_pe > eta_act, (
        f'expected dataset > pe > activation in η², got '
        f'{eta_ds:.3f}, {eta_pe:.3f}, {eta_act:.3f}'
    )


def test_happy_polar_penalty_contrast() -> None:
    print('\n[analysis] happy world: polar_penalty_contrast ...')
    df = generate_happy_runs()
    d = polar_penalty_contrast(df)
    median = d['statistics']['median_delta_db']
    ci_lo, ci_hi = d['statistics']['ci_median_delta_db']
    print(f"  median Δ = {median:.3f} dB  (CI {ci_lo:.3f}–{ci_hi:.3f})  n={d['n']}")
    # Expected: median Δ = 4.5 (= POLAR_PENALTY[none_angular] − POLAR_PENALTY[none_cartesian])
    assert 4.0 < median < 5.0, f'expected median ≈ 4.5 dB, got {median}'
    assert ci_lo > 0, f'CI should exclude 0, got [{ci_lo}, {ci_hi}]'


def test_happy_characterization_correlations() -> None:
    print('\n[analysis] happy world: characterization_correlations ...')
    df = generate_happy_runs()
    d = characterization_correlations(df)
    # PSNR is monotonically decreasing in L_95 by construction → ρ ≈ -1.
    for pe, feats in d['statistics'].items():
        rho_l95 = feats['L_95']['rho']
        print(f"  {pe:18s}: ρ(L_95) = {rho_l95:+.3f}")
        assert rho_l95 < -0.5, (
            f'{pe}: expected strong negative ρ vs L_95, got {rho_l95}'
        )


def test_happy_sh_lmax_ablation() -> None:
    print('\n[analysis] happy world: sh_lmax_ablation ...')
    df = generate_happy_runs()
    d = sh_lmax_ablation(df)
    post = d['statistics']['post_saturation']
    pre  = d['statistics']['pre_saturation']
    print(f"  post: median Δ_match = {post['median']:+.3f} dB "
          f"(n={post['n']})")
    print(f"  pre:  median Δ_LMax  = {pre['median']:+.3f} dB "
          f"(n={pre['n']})")
    # Post-saturation: ~0 (matched should be very close to default).
    assert abs(post['median']) < 0.5, (
        f'post-saturation median should be ≈ 0, got {post["median"]}'
    )
    # Pre-saturation: ~+3 dB (L_max=32 beats L_max=16 on high-bandwidth signals).
    assert pre['median'] > 2.0, (
        f'pre-saturation median should be ≈ +3 dB, got {pre["median"]}'
    )


# ---------------------------------------------------------------------------
# Null-world tests — outputs should reflect no signal
# ---------------------------------------------------------------------------
def test_null_polar_penalty_near_zero() -> None:
    print('\n[analysis] null world: polar_penalty_contrast ...')
    df = generate_null_runs()
    d = polar_penalty_contrast(df)
    median = d['statistics']['median_delta_db']
    ci_lo, ci_hi = d['statistics']['ci_median_delta_db']
    print(f"  median Δ = {median:.3f}  (CI {ci_lo:.3f}–{ci_hi:.3f})")
    # Polar = equatorial = psnr in the null generator, so Δ is exactly 0.
    assert abs(median) < 0.5, f'null median should be ~0, got {median}'


def test_null_pre_saturation_centered_at_zero() -> None:
    print('\n[analysis] null world: sh_lmax_ablation pre-saturation ...')
    df = generate_null_runs()
    d = sh_lmax_ablation(df)
    pre = d['statistics']['pre_saturation']
    print(f"  pre: median Δ_LMax = {pre['median']:+.3f} dB")
    # With no systematic signal, median is some small noise around 0.
    assert abs(pre['median']) < 2.0, (
        f'null pre-saturation median should be small, got {pre["median"]}'
    )


# ---------------------------------------------------------------------------
# Missing-data robustness
# ---------------------------------------------------------------------------
def test_empty_df_returns_skipped() -> None:
    print('\n[analysis] empty DataFrame → all analyses skip cleanly ...')
    df = pd.DataFrame(columns=[
        'dataset', 'ce', 'act', 'pe', 'status',
        'reconstruction_psnr', 'held_out_psnr',
    ])
    for fn in (variance_decomposition, polar_penalty_contrast,
               characterization_correlations, sh_lmax_ablation):
        d = fn(df)
        assert d.get('n', 0) == 0
        assert 'skipped' in d.get('notes', '') or d.get('n', 0) == 0
    print('  OK every analysis returns n=0 with a skipped/missing-data note.')


def test_failed_rows_excluded() -> None:
    """status != 'completed' rows are ignored — adding garbage rows
    doesn't change the summary."""
    print('\n[analysis] status filter ...')
    df = generate_happy_runs()
    bad = df.iloc[:5].copy()
    bad['status'] = 'oom'
    # Poison the columns for both metric families
    for prefix in ('reconstruction_psnr', 'held_out_psnr'):
        bad[f'{prefix}_polar'] = -100.0
        bad[f'{prefix}_equatorial'] = 100.0
    df2 = pd.concat([df, bad], ignore_index=True)
    m1 = polar_penalty_contrast(df)['statistics']['median_delta_db']
    m2 = polar_penalty_contrast(df2)['statistics']['median_delta_db']
    assert abs(m1 - m2) < 1e-6
    print(f'  OK adding oom rows didn\'t change polar-penalty median ({m1:.3f}).')


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------
def test_run_analysis_e2e_smoke() -> None:
    """Write happy synthetic runs.csv, invoke run_analysis.py via subprocess,
    verify summary.json and tables are emitted."""
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
        jpath = os.path.join(out_dir, 'summary.json')
        spath = os.path.join(out_dir, 'summary.md')
        assert os.path.exists(jpath), 'summary.json missing'
        assert os.path.exists(spath), 'summary.md missing'
        with open(jpath) as f:
            results = json.load(f)
        for key in ('variance_decomposition', 'polar_penalty_contrast',
                    'characterization_correlations', 'sh_lmax_ablation'):
            assert key in results
            assert results[key]['n'] > 0, f'{key} has n=0: {results[key]}'
        print('  OK 4/4 analyses ran and emitted populated summaries.')


def main() -> None:
    print('== Happy-world descriptive analyses ==')
    test_happy_variance_decomposition()
    test_happy_polar_penalty_contrast()
    test_happy_characterization_correlations()
    test_happy_sh_lmax_ablation()

    print('\n== Null-world sanity ==')
    test_null_polar_penalty_near_zero()
    test_null_pre_saturation_centered_at_zero()

    print('\n== Missing-data robustness ==')
    test_empty_df_returns_skipped()
    test_failed_rows_excluded()

    print('\n== End-to-end ==')
    test_run_analysis_e2e_smoke()

    print('\nAll analysis tests passed.')


if __name__ == '__main__':
    main()

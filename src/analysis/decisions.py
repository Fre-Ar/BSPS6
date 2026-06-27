"""
Per-analysis descriptive evaluators (preregistration §2).

Each `<analysis_name>(df, ...)` consumes a pandas DataFrame containing the
runs.csv rows for the relevant cells and returns a `dict` of the form

    {
        'name':       <str>,                  # short analysis label
        'summary':    <str>,                  # human-readable summary string
        'statistics': {<key>: <value>, ...},  # numerical summaries
        'data':       <dict-of-arrays>,       # raw per-cell data for figures
        'n':          <int>,                  # sample size used
        'notes':      <str>,                  # caveats / skipped messages
    }

There is NO `decision` field. The analyses are pre-committed descriptive
procedures, not hypothesis tests; the output is a numerical summary that
gets reported as-is, regardless of magnitude or direction. See
preregistration §2 for the rationale.

Statistical tools:
  * Wilcoxon signed-rank — scipy.stats.wilcoxon (reported as a diagnostic
    in §2.2 / §2.4, not as a pass/fail gate).
  * Spearman correlation — scipy.stats.spearmanr.
  * Bootstrap CIs — numpy resampling.
  * Variance decomposition via plain η² (SS_factor / SS_total).

Factor structure (preregistration §3.2 / §3.3):
  * activations: ReLU, ScaledSine, Gaussian
  * PE cells:    none_angular, none_cartesian, rff, sh, fkan
The (activation × PE) Coordinate-MLP cells × the 5 datasets are the main
analysis set. SH cells are additionally evaluated at L_max ∈ {16, ⌈L_95⌉}
for §2.4.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .characterization import DATASET_METRICS, FEATURE_NAMES


# ----- Constants ------------------------------------------------------------
LOW_BANDWIDTH_DATASETS = ('era5', 'hdri_sky')                # L_95 < 32
HIGH_BANDWIDTH_DATASETS = ('etopo1', 'hdri_urban', 'cmb')    # L_95 > 32
ALL_DATASETS = tuple(DATASET_METRICS.keys())

# SH L_max values used in the grid (matches scripts/run_grid.py).
SH_LMAX_DEFAULT = 32
SH_LMAX_H5A = {'era5': 13, 'hdri_sky': 31}
SH_LMAX_H5B = 16

# Activation CLI flag values → cell-key activation tokens.
ACT_FLAG_TO_KEY = {
    'relu':         'relu',
    'scaled-sine':  'scaled_sine',
    'gaussian':     'gaussian',
}

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_RNG_SEED = 42


# ============================================================================
# Cell-selection helpers
# ============================================================================
def _parse_lmax(json_str: str) -> Optional[int]:
    """Pull L_max out of an encoding_kwargs_json string."""
    try:
        d = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    val = d.get('L_max')
    return None if val is None else int(val)


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    """Only rows with status='completed'."""
    if 'status' not in df.columns:
        return df
    return df[df['status'] == 'completed'].copy()


def _pe_cell_key(ce: str, pe: str) -> str:
    """Map (ce, pe) → PE cell-key (see config/architectures.py PE_CELLS)."""
    table = {
        ('angular',              'None'): 'none_angular',
        ('cartesian',            'None'): 'none_cartesian',
        ('cartesian',            'RFF'):  'rff',
        ('spherical-harmonics',  'None'): 'sh',
        ('cartesian',            'FKAN'): 'fkan',
    }
    return table.get((ce, pe), f'{ce}_{pe}')


def _add_cell_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Augment df with derived 'pe_cell' and 'activation' columns. Copy."""
    df = df.copy()
    df['pe_cell'] = [_pe_cell_key(c, p) for c, p in zip(df['ce'], df['pe'])]
    df['activation'] = df['act'].map(ACT_FLAG_TO_KEY).fillna(df['act'])
    return df


def _main_grid_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Completed main-grid cells (SH cells filtered to L_max=SH_LMAX_DEFAULT)."""
    df = _completed(df)
    df = _add_cell_columns(df)
    sh_mask = df['pe_cell'] == 'sh'
    if 'encoding_kwargs_json' in df.columns:
        lmax = df['encoding_kwargs_json'].apply(_parse_lmax)
        keep = (~sh_mask) | (sh_mask & (lmax == SH_LMAX_DEFAULT))
        return df[keep].copy()
    return df


def _sh_cells_at_lmax(df: pd.DataFrame, lmax: int) -> pd.DataFrame:
    """SH rows with the given L_max (used by §2.4 sub-grid analyses)."""
    df = _completed(df)
    df = _add_cell_columns(df)
    if 'encoding_kwargs_json' not in df.columns:
        return df.iloc[0:0].copy()
    parsed = df['encoding_kwargs_json'].apply(_parse_lmax)
    mask = (df['pe_cell'] == 'sh') & (parsed == lmax)
    return df[mask].copy()


# ============================================================================
# Generic helpers
# ============================================================================
def _bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_RNG_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% bootstrap CI of the sample median by row-resampling with replacement."""
    if values.size == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = values.size
    medians = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        medians[i] = float(np.median(values[idx]))
    return (
        float(np.percentile(medians, 100 * alpha / 2)),
        float(np.percentile(medians, 100 * (1 - alpha / 2))),
    )


def _missing_data(name: str, missing: set) -> dict:
    return {
        'name': name,
        'summary': f'Insufficient data; missing: {sorted(missing)}',
        'statistics': {},
        'data': {},
        'n': 0,
        'notes': 'skipped due to missing data',
    }


# ============================================================================
# 2.1 Variance decomposition (PE vs. activation vs. dataset)
# ============================================================================
def _eta_squared(values: np.ndarray, factors: dict[str, np.ndarray]) -> dict[str, float]:
    """Plain η² = SS_factor / SS_total for each factor in `factors`.
    Bounded in [0, 1] for one-factor designs; sums can exceed 1 in unbalanced
    or bootstrap-resampled designs (we do not clamp)."""
    n = values.size
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    out: dict[str, float] = {}
    for name, codes in factors.items():
        levels = pd.unique(codes)
        ss = 0.0
        for lvl in levels:
            mask = codes == lvl
            n_l = int(mask.sum())
            if n_l == 0:
                continue
            ss += n_l * (values[mask].mean() - grand_mean) ** 2
        out[name] = float(ss / ss_total) if ss_total > 0 else 0.0
    return out


def _bootstrap_eta_and_diffs(
    values: np.ndarray,
    factors: dict[str, np.ndarray],
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_RNG_SEED,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Bootstrap 95% CIs for each factor's η² and for the three pairwise diffs.

    Returns (per_factor_ci, diff_ci). diff_ci keys are
    'pe_minus_activation', 'pe_minus_dataset', 'activation_minus_dataset'.
    """
    rng = np.random.default_rng(seed)
    n = values.size
    factor_names = list(factors.keys())
    samples: dict[str, list[float]] = {k: [] for k in factor_names}
    diff_samples: dict[str, list[float]] = {
        'pe_minus_activation':       [],
        'pe_minus_dataset':          [],
        'activation_minus_dataset':  [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_factors = {k: v[idx] for k, v in factors.items()}
        eta = _eta_squared(values[idx], boot_factors)
        for k in factor_names:
            samples[k].append(eta.get(k, 0.0))
        diff_samples['pe_minus_activation'].append(
            eta.get('pe', 0.0) - eta.get('activation', 0.0)
        )
        diff_samples['pe_minus_dataset'].append(
            eta.get('pe', 0.0) - eta.get('dataset', 0.0)
        )
        diff_samples['activation_minus_dataset'].append(
            eta.get('activation', 0.0) - eta.get('dataset', 0.0)
        )

    def _ci(vals: list[float]) -> tuple[float, float]:
        arr = np.asarray(vals)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return (
        {k: _ci(samples[k]) for k in factor_names},
        {k: _ci(diff_samples[k]) for k in diff_samples},
    )


def variance_decomposition(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """§2.1 — Decompose PSNR variance over {pe, activation, dataset}.

    Reports per-factor η² with bootstrap CIs and the three pairwise difference
    CIs. No accept/reject; the magnitudes are the finding.
    """
    if not {metric, 'ce', 'pe', 'act', 'dataset'}.issubset(df.columns):
        missing = {metric, 'ce', 'pe', 'act', 'dataset'} - set(df.columns)
        return _missing_data('variance_decomposition', missing)
    sub = _main_grid_cells(df)
    sub = sub.dropna(subset=[metric]).copy()
    if len(sub) < 8:
        return _missing_data('variance_decomposition', {'enough_cells'})

    values = sub[metric].to_numpy(dtype=float)
    factors = {
        'pe':         sub['pe_cell'].to_numpy(),
        'activation': sub['activation'].to_numpy(),
        'dataset':    sub['dataset'].to_numpy(),
    }
    eta = _eta_squared(values, factors)
    per_factor_ci, diff_ci = _bootstrap_eta_and_diffs(values, factors)

    summary = (
        f"η²: pe={eta['pe']:.3f} "
        f"(CI {per_factor_ci['pe'][0]:.3f}–{per_factor_ci['pe'][1]:.3f}), "
        f"activation={eta['activation']:.3f} "
        f"(CI {per_factor_ci['activation'][0]:.3f}–{per_factor_ci['activation'][1]:.3f}), "
        f"dataset={eta['dataset']:.3f} "
        f"(CI {per_factor_ci['dataset'][0]:.3f}–{per_factor_ci['dataset'][1]:.3f})."
    )

    return {
        'name': '2.1 variance decomposition (PE vs activation vs dataset)',
        'summary': summary,
        'statistics': {
            'eta_sq_pe':         eta['pe'],
            'eta_sq_activation': eta['activation'],
            'eta_sq_dataset':    eta['dataset'],
            'ci_pe':             per_factor_ci['pe'],
            'ci_activation':     per_factor_ci['activation'],
            'ci_dataset':        per_factor_ci['dataset'],
            'diff_ci_pe_minus_activation':      diff_ci['pe_minus_activation'],
            'diff_ci_pe_minus_dataset':         diff_ci['pe_minus_dataset'],
            'diff_ci_activation_minus_dataset': diff_ci['activation_minus_dataset'],
        },
        'n': int(values.size),
        'data': {
            'eta_sq': eta,
            'ci':     {k: list(v) for k, v in per_factor_ci.items()},
            'diff_ci': {k: list(v) for k, v in diff_ci.items()},
        },
        'notes': '',
    }


# ============================================================================
# 2.2 Polar-penalty contrast (none_angular vs none_cartesian)
# ============================================================================
def polar_penalty_contrast(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """§2.2 — Paired difference of polar penalty (none_angular − none_cartesian),
    paired by (activation, dataset)."""
    polar_col = f'{metric}_polar'
    eq_col = f'{metric}_equatorial'
    needed = {polar_col, eq_col, 'ce', 'pe', 'dataset', 'act'}
    if missing := (needed - set(df.columns)):
        return _missing_data('polar_penalty_contrast', missing)

    main = _main_grid_cells(df)
    if len(main) == 0:
        return _missing_data('polar_penalty_contrast', {'main_grid_cells'})

    main['polar_penalty'] = main[eq_col] - main[polar_col]
    pivot = main.pivot_table(
        index=['activation', 'dataset'],
        columns='pe_cell',
        values='polar_penalty',
        aggfunc='first',
    )
    if 'none_angular' not in pivot.columns or 'none_cartesian' not in pivot.columns:
        return _missing_data(
            'polar_penalty_contrast', {'none_angular', 'none_cartesian'},
        )
    delta = (pivot['none_angular'] - pivot['none_cartesian']).dropna()
    if len(delta) < 2:
        return _missing_data('polar_penalty_contrast', {'paired_observations'})

    median = float(delta.median())
    ci_lo, ci_hi = _bootstrap_median_ci(delta.values)

    # Wilcoxon as a diagnostic only (NOT a pass/fail criterion). Guard against
    # the all-zero case (the null world's signature).
    if np.allclose(delta.values, 0.0, atol=1e-12):
        wstat = 0.0
        pvalue = 1.0
    else:
        try:
            w = stats.wilcoxon(delta.values, alternative='greater',
                               zero_method='wilcox')
            wstat = float(w.statistic)
            pvalue = float(w.pvalue)
        except ValueError:
            wstat = float('nan')
            pvalue = float('nan')

    summary = (
        f'median Δ(none_angular − none_cartesian) = {median:.3f} dB '
        f'(95% bootstrap CI {ci_lo:.3f}–{ci_hi:.3f}; n={len(delta)}). '
        f'Wilcoxon W = {wstat:.2f}, p = {pvalue:.4f} (diagnostic only).'
    )

    return {
        'name': '2.2 polar-penalty contrast (none_angular vs none_cartesian)',
        'summary': summary,
        'statistics': {
            'median_delta_db':    median,
            'ci_median_delta_db': (ci_lo, ci_hi),
            'wilcoxon_W':         wstat,
            'wilcoxon_p_value':   pvalue,
        },
        'n': int(len(delta)),
        'data': {
            'delta_per_activation_dataset': delta.to_dict(),
        },
        'notes': '',
    }


# ============================================================================
# 2.3 Characterization correlations (Spearman per PE cell × feature)
# ============================================================================
def _bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_RNG_SEED,
) -> tuple[float, float]:
    """95% bootstrap CI of Spearman ρ by resampling (x, y) pairs."""
    if x.size < 3 or y.size < 3:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = x.size
    rhos = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        # If all xi or all yi happen to coincide, Spearman is undefined; skip.
        if np.all(xi == xi[0]) or np.all(yi == yi[0]):
            rhos[i] = np.nan
            continue
        rho_b, _ = stats.spearmanr(xi, yi)
        rhos[i] = rho_b
    rhos = rhos[~np.isnan(rhos)]
    if rhos.size == 0:
        return (float('nan'), float('nan'))
    return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))


def characterization_correlations(
    df: pd.DataFrame,
    metric: str = 'held_out_psnr',
) -> dict:
    """§2.3 — Spearman rank correlation of per-dataset mean PSNR vs each of
    (L_95, CV, P99_norm), computed per PE cell. 5 PEs × 3 features = 15
    correlations + bootstrap CIs + uncorrected Spearman p-values."""
    needed = {metric, 'ce', 'pe', 'act', 'dataset'}
    if missing := (needed - set(df.columns)):
        return _missing_data('characterization_correlations', missing)
    sub = _main_grid_cells(df)
    sub = sub.dropna(subset=[metric]).copy()
    if len(sub) < 8:
        return _missing_data('characterization_correlations', {'enough_cells'})

    # Mean PSNR over activations per (pe_cell, dataset).
    grouped = (
        sub.groupby(['pe_cell', 'dataset'])[metric].mean().reset_index()
    )

    per_pe: dict[str, dict[str, dict]] = {}
    summary_lines: list[str] = []

    for pe_cell in sorted(grouped['pe_cell'].unique()):
        cell_df = grouped[grouped['pe_cell'] == pe_cell]
        cell_data: dict[str, dict] = {}
        for feature in FEATURE_NAMES:
            feat_vals = np.array(
                [DATASET_METRICS[d][feature] for d in cell_df['dataset']],
                dtype=float,
            )
            psnr_vals = cell_df[metric].to_numpy(dtype=float)
            if feat_vals.size < 3:
                cell_data[feature] = {
                    'rho': float('nan'), 'p_value': float('nan'),
                    'ci_lo': float('nan'), 'ci_hi': float('nan'),
                    'n': int(feat_vals.size),
                    'note': 'insufficient n for Spearman',
                }
                continue
            rho, pvalue = stats.spearmanr(feat_vals, psnr_vals)
            ci_lo, ci_hi = _bootstrap_spearman_ci(feat_vals, psnr_vals)
            cell_data[feature] = {
                'rho': float(rho), 'p_value': float(pvalue),
                'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
                'n': int(feat_vals.size),
                'datasets':  cell_df['dataset'].tolist(),
                'feature_values': feat_vals.tolist(),
                'psnr_values':    psnr_vals.tolist(),
                'note': '',
            }
            summary_lines.append(
                f'  {pe_cell:>15s} / {feature:>9s}: ρ={float(rho):+.3f} '
                f'(CI {float(ci_lo):+.3f}–{float(ci_hi):+.3f}, '
                f'p={float(pvalue):.3f})'
            )
        per_pe[pe_cell] = cell_data

    summary = (
        f'Spearman ρ(per-dataset mean PSNR, feature) for {len(per_pe)} PE '
        f'cells × {len(FEATURE_NAMES)} features:\n' + '\n'.join(summary_lines)
    )

    return {
        'name': '2.3 characterization correlations (Spearman ρ per PE cell × feature)',
        'summary': summary,
        'statistics': {
            pe: {feat: {'rho': info['rho'], 'p_value': info['p_value'],
                        'ci': (info['ci_lo'], info['ci_hi'])}
                 for feat, info in feats.items()}
            for pe, feats in per_pe.items()
        },
        'n': int(grouped['dataset'].nunique()),    # = 5 (datasets)
        'data': {'per_pe_cell': per_pe},
        'notes': (
            'n=5 datasets per correlation. Spearman ρ is rank-only — '
            'we make no linearity claim. Sign convention: positive ρ means '
            'higher feature value associates with higher PSNR. See '
            'preregistration §2.3 for the theoretical sign priors.'
        ),
    }


# ============================================================================
# 2.4 SH L_max ablation (post- and pre-saturation regimes)
# ============================================================================
def _sh_lmax_paired_deltas(
    df: pd.DataFrame,
    default_lmax: int,
    other_lmax_per_dataset: dict[str, int],
    metric: str,
    sign: int,
) -> list[tuple[str, str, int, float]]:
    """Helper: list of (activation, dataset, other_lmax, Δ) tuples where
    Δ = sign · (PSNR(other_lmax) − PSNR(default_lmax))."""
    default_sh = _sh_cells_at_lmax(df, default_lmax)
    deltas: list[tuple[str, str, int, float]] = []
    for ds, other_lmax in other_lmax_per_dataset.items():
        other = _sh_cells_at_lmax(df, other_lmax)
        a = other[other['dataset'] == ds]
        b = default_sh[default_sh['dataset'] == ds]
        if len(a) == 0 or len(b) == 0:
            continue
        merged = a.merge(b, on=['activation'],
                         suffixes=('_other', '_default'))
        for _, row in merged.iterrows():
            mp = row.get(f'{metric}_other')
            dp = row.get(f'{metric}_default')
            if mp is None or dp is None or pd.isna(mp) or pd.isna(dp):
                continue
            deltas.append(
                (row['activation'], ds, int(other_lmax),
                 sign * float(mp - dp))
            )
    return deltas


def _summarize_deltas(label: str, deltas: list[tuple],
                      delta_idx: int) -> dict:
    """Common summary: median + bootstrap CI + diagnostic Wilcoxon."""
    if len(deltas) < 2:
        return _missing_data(label, {'paired_observations'})
    delta_values = np.array([d[delta_idx] for d in deltas], dtype=float)
    median = float(np.median(delta_values))
    ci_lo, ci_hi = _bootstrap_median_ci(delta_values)
    if np.allclose(delta_values, 0.0, atol=1e-12):
        wstat, pvalue = 0.0, 1.0
    else:
        try:
            w = stats.wilcoxon(delta_values, alternative='greater',
                               zero_method='wilcox')
            wstat, pvalue = float(w.statistic), float(w.pvalue)
        except ValueError:
            wstat, pvalue = float('nan'), float('nan')
    return {
        'median':            median,
        'ci_median':         (ci_lo, ci_hi),
        'wilcoxon_W':        wstat,
        'wilcoxon_p_value':  pvalue,
        'delta_values':      delta_values.tolist(),
        'n':                 int(delta_values.size),
    }


def sh_lmax_ablation(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """§2.4 — Paired SH L_max effects in two regimes.

    Post-saturation (low-bandwidth datasets, L_95 < 32):
        Δ_match = PSNR(L_max=⌈L_95⌉) − PSNR(L_max=32).
    Pre-saturation (high-bandwidth datasets, L_95 > 32):
        Δ_LMax = PSNR(L_max=32) − PSNR(L_max=16).

    Reports per-regime median + bootstrap CI + diagnostic Wilcoxon, plus
    the full per-cell delta arrays for plotting.
    """
    # Post-saturation regime
    post_deltas = _sh_lmax_paired_deltas(
        df, default_lmax=SH_LMAX_DEFAULT,
        other_lmax_per_dataset=SH_LMAX_H5A,
        metric=metric, sign=+1,
    )
    post = _summarize_deltas('sh_lmax_ablation/post_saturation', post_deltas,
                             delta_idx=3)

    # Pre-saturation regime
    pre_deltas = _sh_lmax_paired_deltas(
        df, default_lmax=SH_LMAX_DEFAULT,
        other_lmax_per_dataset={ds: SH_LMAX_H5B for ds in HIGH_BANDWIDTH_DATASETS},
        metric=metric, sign=-1,  # PSNR(L_max=32) − PSNR(L_max=16)
    )
    pre = _summarize_deltas('sh_lmax_ablation/pre_saturation', pre_deltas,
                            delta_idx=3)

    summary = (
        f"Post-saturation Δ_match (L_max=⌈L_95⌉ − L_max=32): "
        f"median={post.get('median', float('nan')):.3f} dB "
        f"(CI {post.get('ci_median', (float('nan'),)*2)[0]:.3f}"
        f"–{post.get('ci_median', (float('nan'),)*2)[1]:.3f}, "
        f"n={post.get('n', 0)}). "
        f"Pre-saturation Δ_LMax (L_max=32 − L_max=16): "
        f"median={pre.get('median', float('nan')):.3f} dB "
        f"(CI {pre.get('ci_median', (float('nan'),)*2)[0]:.3f}"
        f"–{pre.get('ci_median', (float('nan'),)*2)[1]:.3f}, "
        f"n={pre.get('n', 0)})."
    )

    return {
        'name': '2.4 SH L_max ablation (post- and pre-saturation regimes)',
        'summary': summary,
        'statistics': {
            'post_saturation': {k: v for k, v in post.items()
                                if k not in ('delta_values',)},
            'pre_saturation':  {k: v for k, v in pre.items()
                                if k not in ('delta_values',)},
        },
        'n': int(post.get('n', 0)) + int(pre.get('n', 0)),
        'data': {
            'post_saturation_deltas': [
                {'activation': a, 'dataset': d, 'lmax_matched': lmax, 'delta_db': v}
                for (a, d, lmax, v) in post_deltas
            ],
            'pre_saturation_deltas': [
                {'activation': a, 'dataset': d, 'delta_db': v}
                for (a, d, _lmax, v) in pre_deltas
            ],
        },
        'notes': (
            'Wilcoxon p-values are diagnostics, not pass/fail criteria. '
            'At n=6 (post-saturation) the smallest achievable two-sided p '
            'is ~0.031; at n=9 (pre-saturation) it is ~0.004.'
        ),
    }

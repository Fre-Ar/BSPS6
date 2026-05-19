"""
Per-hypothesis evaluators.

Each `evaluate_hN(df, ...)` consumes a pandas DataFrame containing the
runs.csv rows for the relevant cells and returns a `dict` of the form

    {
        'name':       <str>,                  # short hypothesis label
        'decision':   'accepted' | 'rejected' | 'inconclusive',
        'reasoning':  <str>,                  # human-readable summary
        'statistic':  {<name>: <value>, ...}, # observed test statistic(s)
        'threshold':  {<name>: <value>, ...}, # pre-committed thresholds
        'n':          <int>,                  # sample size used
        'data':       <dict-of-arrays>,       # raw paired/grouped data for figures
        'notes':      <str>,                  # caveats or skipped-due-to-missing-data
    }

Decisions are STRICTLY governed by the pre-committed thresholds in
docs/preregistration.md. The evaluator code is the single source of truth
for *how* the thresholds are applied; the thresholds themselves are
hard-coded here so that drift requires a preregistration amendment.

Statistical tools:
  * Paired Wilcoxon — scipy.stats.wilcoxon
  * Spearman correlation — scipy.stats.spearmanr
  * OLS regression — manual (X^T X)^-1 X^T y in numpy, no sklearn dep
  * Bootstrap CIs — numpy resampling

Notes on the (effective) single-seed protocol: every (architecture, encoding,
dataset, encoding_kwargs) combination has at most one row in the DataFrame
(one seed). Hypothesis sample sizes are correspondingly small.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .characterization import DATASET_METRICS, FEATURE_NAMES


# ----- Constants ------------------------------------------------------------
# H4 / H5a / H5b dataset partitions.
LOW_BANDWIDTH_DATASETS = ('era5', 'hdri_sky')           # L_95 < 32 → H5a
HIGH_BANDWIDTH_DATASETS = ('etopo1', 'hdri_urban', 'cmb')  # L_95 > 32 → H5b
ALL_DATASETS = tuple(DATASET_METRICS.keys())

# SH L_max values used in the grid (matches scripts/run_grid.py).
SH_LMAX_DEFAULT = 32
SH_LMAX_H5A = {'era5': 13, 'hdri_sky': 31}
SH_LMAX_H5B = 16

# Default thresholds — pre-committed in preregistration.
H1_ACCEPT_EFFECT_DB = 1.0
H1_ACCEPT_P = 0.01
H1_REJECT_EFFECT_DB = 0.3

H5_1_ACCEPT_RHO = 0.5
H5_1_ACCEPT_P = 0.05
H5_1_REJECT_ABS_RHO = 0.2
H5_1_REJECT_P = 0.20

H3_ACCEPT_RATIO = 2.0
H3_REJECT_RATIO = 1.0

H4_ACCEPT_R2 = 0.5
H4_REJECT_R2 = 0.2
H4_ACCEPT_MIN_ENCODINGS = 3
H4_REJECT_MIN_ENCODINGS = 3

H5A_ACCEPT_MARGIN_DB = -0.5
H5A_REJECT_MARGIN_DB = -1.5

H5B_ACCEPT_DELTA_DB = 0.5
H5B_REJECT_DELTA_DB = 0.0

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_RNG_SEED = 42


# ============================================================================
# Cell-selection helpers
# ============================================================================
def _parse_lmax(json_str: str) -> Optional[int]:
    """Pull L_max out of an encoding_kwargs_json string. Returns None if the
    column isn't an SH kwargs object."""
    try:
        d = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    val = d.get('L_max')
    return None if val is None else int(val)


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    """Only rows with status='completed'. Other rows (oom/nan/etc.) are
    excluded from every analysis."""
    if 'status' not in df.columns:
        return df
    return df[df['status'] == 'completed'].copy()


def _main_grid_cells(df: pd.DataFrame) -> pd.DataFrame:
    """The 80 main-grid cells. For SH, only the L_max=32 cells qualify."""
    df = _completed(df)
    sh_mask = df['ce'] == 'spherical-harmonics'
    if 'encoding_kwargs_json' in df.columns:
        lmax = df['encoding_kwargs_json'].apply(_parse_lmax)
        # Keep all non-SH rows + SH rows with L_max=SH_LMAX_DEFAULT.
        keep = (~sh_mask) | (sh_mask & (lmax == SH_LMAX_DEFAULT))
        return df[keep].copy()
    return df


def _sh_cells_at_lmax(df: pd.DataFrame, lmax: int) -> pd.DataFrame:
    """SH rows with the given L_max."""
    df = _completed(df)
    if 'encoding_kwargs_json' not in df.columns:
        return df.iloc[0:0].copy()
    parsed = df['encoding_kwargs_json'].apply(_parse_lmax)
    mask = (df['ce'] == 'spherical-harmonics') & (parsed == lmax)
    return df[mask].copy()


# ============================================================================
# H1 — Polar singularities of angular encoding (subsection of H3)
# ============================================================================
def evaluate_h1(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """Angular has higher polar penalty than non-angular encodings.

    Paired Wilcoxon over (arch, dataset) cells, n = 4×5 = 20 paired obs.

    Test variable:
        Δ_i = polar_penalty_angular(i) − mean( polar_penalty_{other}(i) )
    where polar_penalty(i) = equatorial_PSNR(i) − polar_PSNR(i) for cell i.
    """
    df = _main_grid_cells(df)
    polar_col = f'{metric}_polar'
    eq_col = f'{metric}_equatorial'
    needed_cols = {polar_col, eq_col, 'arch', 'ce', 'dataset'}
    if missing := (needed_cols - set(df.columns)):
        return _missing_data_decision('H1', missing)

    # Compute polar penalty per (arch, ce, dataset).
    df = df.copy()
    df['polar_penalty'] = df[eq_col] - df[polar_col]

    # Pivot: rows = (arch, dataset), columns = ce, values = polar_penalty.
    pivot = df.pivot_table(
        index=['arch', 'dataset'],
        columns='ce',
        values='polar_penalty',
        aggfunc='first',
    )
    angular_col = 'angular'
    non_angular = [c for c in pivot.columns if c != angular_col]
    if angular_col not in pivot.columns or not non_angular:
        return _missing_data_decision('H1', {angular_col, 'non_angular_encoding'})

    # For each (arch, dataset), compute Δ.
    other_mean = pivot[non_angular].mean(axis=1)
    delta = pivot[angular_col] - other_mean
    delta = delta.dropna()
    if len(delta) < 2:
        return _missing_data_decision('H1', {'enough_paired_observations'})

    # Wilcoxon signed-rank: H₀ median = 0, alt > 0 (angular polar penalty is
    # bigger). scipy.stats.wilcoxon's `alternative='greater'` tests this.
    median = float(delta.median())
    try:
        wstat = stats.wilcoxon(delta.values, alternative='greater',
                               zero_method='wilcox')
        pvalue = float(wstat.pvalue)
        stat = float(wstat.statistic)
    except ValueError as e:                              # all-zeros corner case
        return _err_decision('H1', f'wilcoxon failed: {e}')

    # Decision
    if median > H1_ACCEPT_EFFECT_DB and pvalue < H1_ACCEPT_P:
        decision = 'accepted'
    elif median < H1_REJECT_EFFECT_DB:
        decision = 'rejected'
    else:
        decision = 'inconclusive'

    return {
        'name': 'H1 — polar singularities (angular > non-angular)',
        'decision': decision,
        'reasoning': (
            f'median Δ(polar penalty) = {median:.3f} dB; '
            f'Wilcoxon W = {stat:.2f}, p = {pvalue:.4f}. '
            f'Accept if median > {H1_ACCEPT_EFFECT_DB} dB and p < {H1_ACCEPT_P}.'
        ),
        'statistic': {
            'median_delta_db': median,
            'wilcoxon_W': stat,
            'p_value': pvalue,
        },
        'threshold': {
            'accept_median_db_gt': H1_ACCEPT_EFFECT_DB,
            'accept_p_lt':         H1_ACCEPT_P,
            'reject_median_db_lt': H1_REJECT_EFFECT_DB,
        },
        'n': int(len(delta)),
        'data': {
            'delta_per_arch_dataset': delta.to_dict(),
        },
        'notes': '',
    }


# ============================================================================
# H5.1 — SH-vs-SRFF crossover (subsection of H5)
# ============================================================================
def evaluate_h5_1(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """SH_advantage(arch, dataset) = PSNR(SH) − PSNR(SRFF); Spearman ρ vs −L_95.

    Datasets are five; per (arch, dataset) under 1 seed we get one
    SH_advantage value → 20 paired observations.
    """
    df = _main_grid_cells(df)
    if metric not in df.columns:
        return _missing_data_decision('H5.1', {metric})

    pivot = df.pivot_table(
        index=['arch', 'dataset'],
        columns='ce',
        values=metric,
        aggfunc='first',
    )
    if 'spherical-harmonics' not in pivot.columns or 'spherical-rff' not in pivot.columns:
        return _missing_data_decision('H5.1', {'spherical-harmonics', 'spherical-rff'})

    sh_adv = (pivot['spherical-harmonics'] - pivot['spherical-rff']).dropna()
    if len(sh_adv) < 3:
        return _missing_data_decision('H5.1', {'enough_paired_observations'})

    # Pair with -L_95 for each (arch, dataset). Architecture is uniform per
    # dataset on L_95, so we just look up by dataset.
    neg_l95 = pd.Series(
        [-DATASET_METRICS[ad[1]]['L_95'] for ad in sh_adv.index],
        index=sh_adv.index,
    )
    rho, pvalue = stats.spearmanr(sh_adv.values, neg_l95.values)
    rho = float(rho)
    pvalue = float(pvalue)

    if rho > H5_1_ACCEPT_RHO and pvalue < H5_1_ACCEPT_P:
        decision = 'accepted'
    elif abs(rho) < H5_1_REJECT_ABS_RHO or pvalue > H5_1_REJECT_P:
        decision = 'rejected'
    else:
        decision = 'inconclusive'

    return {
        'name': 'H5.1 — SH-vs-SRFF crossover (SH wins on low L_95)',
        'decision': decision,
        'reasoning': (
            f'Spearman ρ(SH_advantage, −L_95) = {rho:.3f}, p = {pvalue:.4f}. '
            f'Accept if ρ > {H5_1_ACCEPT_RHO} and p < {H5_1_ACCEPT_P}.'
        ),
        'statistic': {'spearman_rho': rho, 'p_value': pvalue},
        'threshold': {
            'accept_rho_gt': H5_1_ACCEPT_RHO,
            'accept_p_lt':   H5_1_ACCEPT_P,
            'reject_abs_rho_lt': H5_1_REJECT_ABS_RHO,
        },
        'n': int(len(sh_adv)),
        'data': {
            'sh_advantage_db': sh_adv.to_dict(),
            'neg_l95':         neg_l95.to_dict(),
        },
        'notes': '',
    }

# ============================================================================
# H3 — Variance decomposition: encoding vs. architecture vs. dataset
# ============================================================================
def _eta_squared(values: np.ndarray, factors: dict[str, np.ndarray]) -> dict:
    """Compute (plain) η² for each factor in `factors`.

    For a balanced one-factor ANOVA, η²(F) = SS_between(F) / SS_total ∈ [0, 1].
    SS_between(F) = Σ_levels n_level · (mean_level − grand_mean)².

    We compute SS_between per factor independently. For balanced multi-factor
    designs without interactions, the per-factor SS values are additive
    (SS_total = Σ SS_factor + SS_residual); for unbalanced or bootstrap-
    resampled designs they need not be, in which case Σ η²_factor can
    exceed 1. We do NOT clamp — H3's downstream test uses the *difference*
    η²(enc) − η²(arch), which is well-behaved even in degenerate samples.
    """
    n = values.size
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())

    eta_sq: dict[str, float] = {}
    ss_per_factor: dict[str, float] = {}

    for name, codes in factors.items():
        levels = pd.unique(codes)
        ss = 0.0
        for lvl in levels:
            mask = codes == lvl
            n_l = int(mask.sum())
            if n_l == 0:
                continue
            ss += n_l * (values[mask].mean() - grand_mean) ** 2
        ss_per_factor[name] = float(ss)
        eta_sq[name] = float(ss / ss_total) if ss_total > 0 else 0.0

    return {
        'eta_sq': eta_sq,
        'ss_per_factor': ss_per_factor,
        'ss_total': ss_total,
        'n': int(n),
    }


def _bootstrap_eta_sq_and_diff(
    values: np.ndarray, factors: dict[str, np.ndarray],
    diff_factors: tuple[str, str] = ('encoding', 'architecture'),
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_RNG_SEED,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float], list[float]]:
    """95% bootstrap CIs for each factor's η² and for the difference
    η²(diff_factors[0]) − η²(diff_factors[1]). Returns
    (per_factor_ci, diff_ci, diff_samples)."""
    rng = np.random.default_rng(seed)
    n = values.size
    samples: dict[str, list[float]] = {k: [] for k in factors}
    diff_samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_vals = values[idx]
        boot_factors = {k: v[idx] for k, v in factors.items()}
        result = _eta_squared(boot_vals, boot_factors)
        for k, v in result['eta_sq'].items():
            samples[k].append(v)
        a = result['eta_sq'].get(diff_factors[0], 0.0)
        b = result['eta_sq'].get(diff_factors[1], 0.0)
        diff_samples.append(a - b)

    per_factor_ci: dict[str, tuple[float, float]] = {}
    for k, vals in samples.items():
        arr = np.asarray(vals)
        per_factor_ci[k] = (float(np.percentile(arr, 2.5)),
                            float(np.percentile(arr, 97.5)))
    arr_diff = np.asarray(diff_samples)
    diff_ci = (float(np.percentile(arr_diff, 2.5)),
               float(np.percentile(arr_diff, 97.5)))
    return per_factor_ci, diff_ci, diff_samples


def evaluate_h3(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """Variance decomposition: encoding vs architecture vs dataset.

    Tests "encoding dominates over architecture" via two conditions:
      1. Point ratio η²(encoding) / η²(architecture) ≥ 2.0.
      2. 95% bootstrap CI of the difference η²(enc) − η²(arch) excludes 0
         (a stricter and statistically cleaner test than the per-factor
         CI non-overlap criterion in earlier drafts of the preregistration).

    Uses plain η² (SS_factor / SS_total) — bounded in [0, 1] by construction,
    unlike partial η² which can degenerate to 1.0 in bootstrap resamples.
    """
    df = _main_grid_cells(df)
    if metric not in df.columns:
        return _missing_data_decision('H3', {metric})
    df = df.dropna(subset=[metric]).copy()
    if len(df) < 8:
        return _missing_data_decision('H3', {'enough_cells'})

    values = df[metric].to_numpy(dtype=float)
    factors = {
        'encoding':     df['ce'].to_numpy(),
        'architecture': df['arch'].to_numpy(),
        'dataset':      df['dataset'].to_numpy(),
    }
    point = _eta_squared(values, factors)
    per_factor_ci, diff_ci, diff_samples = _bootstrap_eta_sq_and_diff(
        values, factors, diff_factors=('encoding', 'architecture'),
    )

    eta_enc = point['eta_sq']['encoding']
    eta_arch = point['eta_sq']['architecture']
    ratio = (eta_enc / eta_arch) if eta_arch > 0 else float('inf')
    diff_lo, diff_hi = diff_ci
    diff_excludes_zero = diff_lo > 0.0

    if ratio >= H3_ACCEPT_RATIO and diff_excludes_zero:
        decision = 'accepted'
    elif ratio < H3_REJECT_RATIO or eta_arch > eta_enc:
        decision = 'rejected'
    else:
        decision = 'inconclusive'

    return {
        'name': 'H3 — encoding dominates over architecture (variance decomp)',
        'decision': decision,
        'reasoning': (
            f'η²: encoding={eta_enc:.3f} (CI {per_factor_ci["encoding"][0]:.3f}'
            f'–{per_factor_ci["encoding"][1]:.3f}), '
            f'architecture={eta_arch:.3f} '
            f'(CI {per_factor_ci["architecture"][0]:.3f}'
            f'–{per_factor_ci["architecture"][1]:.3f}); '
            f'ratio η²(enc)/η²(arch) = {ratio:.2f}; '
            f'Δη² CI = [{diff_lo:.3f}, {diff_hi:.3f}]. '
            f'Accept if ratio ≥ {H3_ACCEPT_RATIO} and Δη² CI excludes 0.'
        ),
        'statistic': {
            'eta_sq_encoding':     eta_enc,
            'eta_sq_architecture': eta_arch,
            'eta_sq_dataset':      point['eta_sq']['dataset'],
            'ci_encoding':         per_factor_ci['encoding'],
            'ci_architecture':     per_factor_ci['architecture'],
            'ci_dataset':          per_factor_ci['dataset'],
            'ratio_enc_over_arch': ratio,
            'diff_eta_ci':         diff_ci,
            'diff_excludes_zero':  diff_excludes_zero,
        },
        'threshold': {
            'accept_ratio_ge':     H3_ACCEPT_RATIO,
            'accept_diff_lo_gt_0': True,
            'reject_ratio_lt':     H3_REJECT_RATIO,
        },
        'n': point['n'],
        'data': {
            'eta_sq':  point['eta_sq'],
            'ci':      {k: list(v) for k, v in per_factor_ci.items()},
            'diff_ci': list(diff_ci),
        },
        'notes': '',
    }


# ============================================================================
# H4 — Characterization predicts encoding performance
# ============================================================================
def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve (X^T X)^-1 X^T y. Returns coefficient vector."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _build_design_matrix(
    df: pd.DataFrame,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    arch_dummies: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct design matrix (X, y) for the H4 regression.
       Returns X (n × k), y (n,), and column labels."""
    feats: list[np.ndarray] = []
    labels: list[str] = []

    # Intercept
    feats.append(np.ones(len(df)))
    labels.append('intercept')

    # Continuous dataset-level features
    for fname in feature_names:
        col = np.array([DATASET_METRICS[d][fname] for d in df['dataset']],
                       dtype=float)
        feats.append(col)
        labels.append(fname)

    # Architecture dummies (drop one for reference category to avoid singular X).
    if arch_dummies:
        arches = sorted(df['arch'].unique())
        for a in arches[1:]:
            feats.append((df['arch'] == a).to_numpy(dtype=float))
            labels.append(f'arch={a}')

    X = np.column_stack(feats)
    y = df['held_out_psnr'].to_numpy(dtype=float)
    return X, y, labels


def _loo_cv_r2(df: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray]:
    """Leave-one-dataset-out cross-validated R² for the H4 regression.
       Returns (R², y_true, y_pred) where the prediction arrays cover the full
       sample (each row predicted by a model trained without its dataset)."""
    datasets = sorted(df['dataset'].unique())
    preds = np.zeros(len(df))
    true = np.zeros(len(df))
    df = df.reset_index(drop=True)
    for ds in datasets:
        train_idx = df.index[df['dataset'] != ds].to_numpy()
        test_idx = df.index[df['dataset'] == ds].to_numpy()
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_train, y_train, labels = _build_design_matrix(df.iloc[train_idx])
        beta = _ols_fit(X_train, y_train)
        X_test, y_test, _ = _build_design_matrix(df.iloc[test_idx])
        # Align test columns to training columns (arch dummies may differ
        # between folds if a held-out fold has an architecture not present
        # in the training fold — won't happen with 4 archs × 5 datasets but
        # we handle it gracefully).
        if X_test.shape[1] != X_train.shape[1]:
            # Re-build test design matrix using training arch reference set.
            X_test = _align_design_matrix(df.iloc[test_idx], labels)
        preds[test_idx] = X_test @ beta
        true[test_idx] = y_test

    if true.size == 0:
        return float('nan'), np.array([]), np.array([])
    ss_res = float(((true - preds) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return r2, true, preds


def _align_design_matrix(df: pd.DataFrame, training_labels: list[str]) -> np.ndarray:
    """Build a design matrix for `df` using exactly the column ordering of
    `training_labels`. Missing arch columns become all-zeros."""
    cols = []
    for lbl in training_labels:
        if lbl == 'intercept':
            cols.append(np.ones(len(df)))
        elif lbl in FEATURE_NAMES:
            cols.append(np.array(
                [DATASET_METRICS[d][lbl] for d in df['dataset']], dtype=float
            ))
        elif lbl.startswith('arch='):
            target = lbl.split('=', 1)[1]
            cols.append((df['arch'] == target).to_numpy(dtype=float))
        else:
            cols.append(np.zeros(len(df)))
    return np.column_stack(cols)


def _bootstrap_r2_ci(df: pd.DataFrame, n_boot: int = DEFAULT_BOOTSTRAP_N,
                     seed: int = DEFAULT_BOOTSTRAP_RNG_SEED) -> tuple[float, float]:
    """95% bootstrap CI for the LOO-CV R². Resamples (architecture, dataset)
    rows with replacement within each LOO fold."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    boots: list[float] = []
    for _ in range(n_boot):
        # Resample rows preserving the dataset grouping (so every LOO fold
        # still has at least one row).
        sampled_dfs = []
        for ds in sorted(df['dataset'].unique()):
            sub = df[df['dataset'] == ds]
            if len(sub) == 0:
                continue
            idx = rng.integers(0, len(sub), size=len(sub))
            sampled_dfs.append(sub.iloc[idx])
        boot_df = pd.concat(sampled_dfs, ignore_index=True)
        try:
            r2, _, _ = _loo_cv_r2(boot_df)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if math.isfinite(r2):
            boots.append(r2)
    if not boots:
        return (float('nan'), float('nan'))
    arr = np.asarray(boots)
    return (float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)))


def evaluate_h4(df: pd.DataFrame) -> dict:
    """Linear regression of held_out_psnr on (L_95, CV, P99_norm) + arch dummies,
    fit independently per encoding. LOO-CV R² with bootstrap CI."""
    df = _main_grid_cells(df)
    if 'held_out_psnr' not in df.columns:
        return _missing_data_decision('H4', {'held_out_psnr'})
    df = df.dropna(subset=['held_out_psnr']).copy()
    if len(df) < 8:
        return _missing_data_decision('H4', {'enough_cells'})

    per_encoding: dict[str, dict[str, Any]] = {}
    for ce in sorted(df['ce'].unique()):
        sub = df[df['ce'] == ce]
        n_datasets = sub['dataset'].nunique()
        if n_datasets < 2:
            per_encoding[ce] = {
                'r2': float('nan'),
                'ci_lo': float('nan'),
                'ci_hi': float('nan'),
                'n': int(len(sub)),
                'n_datasets': int(n_datasets),
                'note': 'insufficient datasets for LOO-CV',
            }
            continue
        r2, y_true, y_pred = _loo_cv_r2(sub)
        ci_lo, ci_hi = _bootstrap_r2_ci(sub)
        per_encoding[ce] = {
            'r2': float(r2),
            'ci_lo': float(ci_lo),
            'ci_hi': float(ci_hi),
            'n': int(len(sub)),
            'n_datasets': int(n_datasets),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'note': '',
        }

    # Decision is driven by the point R² alone. The bootstrap CI is reported
    # for transparency but does NOT gate acceptance: at n=5 datasets with a
    # wide L_95 range (13–236), LOO-CV R² is fundamentally noisy when the
    # held-out fold is the outlier — see preregistration §5 "Limitations".
    n_pass = sum(
        1 for r in per_encoding.values()
        if math.isfinite(r['r2']) and r['r2'] > H4_ACCEPT_R2
    )
    n_fail = sum(
        1 for r in per_encoding.values()
        if math.isfinite(r['r2']) and r['r2'] < H4_REJECT_R2
    )

    if n_pass >= H4_ACCEPT_MIN_ENCODINGS:
        decision = 'accepted'
    elif n_fail >= H4_REJECT_MIN_ENCODINGS:
        decision = 'rejected'
    else:
        decision = 'inconclusive'

    return {
        'name': 'H4 — characterization predicts encoding performance',
        'decision': decision,
        'reasoning': (
            f'{n_pass}/{len(per_encoding)} encodings have LOO-CV R² > '
            f'{H4_ACCEPT_R2} (point estimate). '
            f'Accept threshold: ≥{H4_ACCEPT_MIN_ENCODINGS} encodings. '
            f'Bootstrap CIs reported for transparency but do not gate '
            f'the decision (see preregistration §5).'
        ),
        'statistic': {
            ce: {'r2': r['r2'], 'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi']}
            for ce, r in per_encoding.items()
        },
        'threshold': {
            'accept_r2_gt': H4_ACCEPT_R2,
            'accept_min_encodings': H4_ACCEPT_MIN_ENCODINGS,
            'reject_r2_lt': H4_REJECT_R2,
            'reject_min_encodings': H4_REJECT_MIN_ENCODINGS,
        },
        'n': int(sum(r['n'] for r in per_encoding.values())),
        'data': {'per_encoding': per_encoding},
        'notes': '',
    }


# ============================================================================
# H5a — Post-saturation tail (over-shooting L_max wastes parameters)
# ============================================================================
def evaluate_h5a(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """For datasets with L_95 < 32: PSNR(matched L_max) within 0.5 dB of L_max=32.

    Δ_match = PSNR(matched) − PSNR(L_max=32)
    Wilcoxon: H₀ median ≥ −0.5 dB.
    """
    main_sh = _sh_cells_at_lmax(df, SH_LMAX_DEFAULT)
    deltas: list[tuple] = []
    for ds, lmax_matched in SH_LMAX_H5A.items():
        matched = _sh_cells_at_lmax(df, lmax_matched)
        m = matched[matched['dataset'] == ds]
        b = main_sh[main_sh['dataset'] == ds]
        if len(m) == 0 or len(b) == 0:
            continue
        merged = m.merge(b, on=['arch'], suffixes=('_matched', '_default'))
        for _, row in merged.iterrows():
            mp = row.get(f'{metric}_matched')
            dp = row.get(f'{metric}_default')
            if mp is None or dp is None or pd.isna(mp) or pd.isna(dp):
                continue
            deltas.append((row['arch'], ds, lmax_matched, float(mp - dp)))

    if len(deltas) < 3:
        return _missing_data_decision('H5a', {'enough_paired_observations'})

    delta_values = np.array([d[3] for d in deltas])
    median = float(np.median(delta_values))
    try:
        # H₀: median ≥ -0.5 dB. Equivalently, test if delta - (-0.5) ≥ 0,
        # i.e., delta + 0.5 ≥ 0. Wilcoxon with alternative='greater'.
        shifted = delta_values + 0.5
        # If all shifted values are identical, wilcoxon raises; handle gracefully.
        if np.allclose(shifted, shifted[0]):
            pvalue = 0.5
            wstat = 0.0
        else:
            wresult = stats.wilcoxon(shifted, alternative='greater',
                                     zero_method='wilcox')
            pvalue = float(wresult.pvalue)
            wstat = float(wresult.statistic)
    except ValueError as e:
        return _err_decision('H5a', f'wilcoxon failed: {e}')

    if median > H5A_ACCEPT_MARGIN_DB and pvalue < 0.05:
        decision = 'accepted'
    elif median < H5A_REJECT_MARGIN_DB:
        decision = 'rejected'
    else:
        decision = 'inconclusive'

    return {
        'name': 'H5a — post-saturation tail (matched L_max retains PSNR)',
        'decision': decision,
        'reasoning': (
            f'median Δ_match = {median:.3f} dB (n={len(delta_values)}); '
            f'Wilcoxon W = {wstat:.2f}, p = {pvalue:.4f}. '
            f'Accept if median > {H5A_ACCEPT_MARGIN_DB} dB and p < 0.05.'
        ),
        'statistic': {
            'median_delta_db': median,
            'wilcoxon_W': wstat,
            'p_value': pvalue,
        },
        'threshold': {
            'accept_median_db_gt': H5A_ACCEPT_MARGIN_DB,
            'accept_p_lt': 0.05,
            'reject_median_db_lt': H5A_REJECT_MARGIN_DB,
        },
        'n': int(len(delta_values)),
        'data': {
            'deltas': [
                {'arch': a, 'dataset': d, 'lmax_matched': l, 'delta_db': v}
                for (a, d, l, v) in deltas
            ],
        },
        'notes': '',
    }


# ============================================================================
# H5b — Pre-saturation slope (under-shooting L_max hurts)
# ============================================================================
def evaluate_h5b(df: pd.DataFrame, metric: str = 'held_out_psnr') -> dict:
    """For datasets with L_95 > 32: PSNR(L_max=32) > PSNR(L_max=16).

    Δ_LMax = PSNR(L_max=32) − PSNR(L_max=16)
    Wilcoxon: H₀ Δ ≤ 0; reject if median Δ > 0.5 dB at p < 0.05.
    """
    main_sh = _sh_cells_at_lmax(df, SH_LMAX_DEFAULT)
    h5b_sh = _sh_cells_at_lmax(df, SH_LMAX_H5B)
    deltas: list[tuple] = []
    for ds in HIGH_BANDWIDTH_DATASETS:
        a = h5b_sh[h5b_sh['dataset'] == ds]
        b = main_sh[main_sh['dataset'] == ds]
        if len(a) == 0 or len(b) == 0:
            continue
        merged = b.merge(a, on=['arch'], suffixes=('_default', '_h5b'))
        for _, row in merged.iterrows():
            dp = row.get(f'{metric}_default')
            hp = row.get(f'{metric}_h5b')
            if dp is None or hp is None or pd.isna(dp) or pd.isna(hp):
                continue
            deltas.append((row['arch'], ds, float(dp - hp)))

    if len(deltas) < 3:
        return _missing_data_decision('H5b', {'enough_paired_observations'})

    delta_values = np.array([d[2] for d in deltas])
    median = float(np.median(delta_values))
    try:
        if np.allclose(delta_values, delta_values[0]):
            pvalue = 0.5
            wstat = 0.0
        else:
            wresult = stats.wilcoxon(delta_values, alternative='greater',
                                     zero_method='wilcox')
            pvalue = float(wresult.pvalue)
            wstat = float(wresult.statistic)
    except ValueError as e:
        return _err_decision('H5b', f'wilcoxon failed: {e}')

    if median > H5B_ACCEPT_DELTA_DB and pvalue < 0.05:
        decision = 'accepted'
    elif median <= H5B_REJECT_DELTA_DB:
        decision = 'rejected'
    else:
        decision = 'inconclusive'

    return {
        'name': 'H5b — pre-saturation slope (more L_max helps when L_95 > 32)',
        'decision': decision,
        'reasoning': (
            f'median Δ_LMax = {median:.3f} dB (n={len(delta_values)}); '
            f'Wilcoxon W = {wstat:.2f}, p = {pvalue:.4f}. '
            f'Accept if median > {H5B_ACCEPT_DELTA_DB} dB and p < 0.05.'
        ),
        'statistic': {
            'median_delta_db': median,
            'wilcoxon_W': wstat,
            'p_value': pvalue,
        },
        'threshold': {
            'accept_median_db_gt': H5B_ACCEPT_DELTA_DB,
            'accept_p_lt': 0.05,
            'reject_median_db_le': H5B_REJECT_DELTA_DB,
        },
        'n': int(len(delta_values)),
        'data': {
            'deltas': [
                {'arch': a, 'dataset': d, 'delta_db': v}
                for (a, d, v) in deltas
            ],
        },
        'notes': '',
    }


# ============================================================================
# Helpers for missing-data / error decisions
# ============================================================================
def _missing_data_decision(name: str, missing: set) -> dict:
    return {
        'name': name,
        'decision': 'inconclusive',
        'reasoning': f'Insufficient data for {name}; missing: {sorted(missing)}',
        'statistic': {},
        'threshold': {},
        'n': 0,
        'data': {},
        'notes': 'skipped due to missing data',
    }


def _err_decision(name: str, msg: str) -> dict:
    return {
        'name': name,
        'decision': 'inconclusive',
        'reasoning': f'Test failed: {msg}',
        'statistic': {},
        'threshold': {},
        'n': 0,
        'data': {},
        'notes': f'error: {msg}',
    }

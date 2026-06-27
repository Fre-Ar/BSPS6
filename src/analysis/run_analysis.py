"""
Main analysis script — consumes results/runs.csv and emits per-analysis
descriptive summaries, markdown tables, and figures (preregistration §6).

Usage from repo root:
    PYTHONPATH=src python src/analysis/run_analysis.py \
        --runs_csv results/runs.csv \
        --output_dir results/analysis

Outputs:
    <output_dir>/summary.json
    <output_dir>/tables/{variance_decomposition,
                         polar_penalty_contrast,
                         characterization_correlations,
                         sh_lmax_ablation}.md
    <output_dir>/figures/{variance_decomposition,
                          polar_penalty_contrast,
                          characterization_correlations,
                          sh_lmax_ablation}.png  (unless --skip_figures)
    <output_dir>/summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Make `src/` importable when invoked directly (PYTHONPATH=src usage).
if __name__ == '__main__':
    SRC_DIR = str(Path(__file__).resolve().parent.parent)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

from analysis.decisions import (                                          # noqa: E402
    variance_decomposition,
    polar_penalty_contrast,
    characterization_correlations,
    sh_lmax_ablation,
)
from analysis.characterization import DATASET_METRICS, FEATURE_NAMES      # noqa: E402


# Pre-committed analysis registry. Order matters for the summary table.
ANALYSES: tuple[tuple[str, callable], ...] = (
    ('variance_decomposition',         variance_decomposition),
    ('polar_penalty_contrast',         polar_penalty_contrast),
    ('characterization_correlations',  characterization_correlations),
    ('sh_lmax_ablation',               sh_lmax_ablation),
)


# ============================================================================
# JSON / markdown writers
# ============================================================================
def _json_safe(obj):
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_summary_json(results: dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_json_safe(results), f, indent=2, sort_keys=False)


def _format_value(v) -> str:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            lo, hi = float(v[0]), float(v[1])
            return f'[{lo:.3f}, {hi:.3f}]'
        except (TypeError, ValueError):
            pass
    if isinstance(v, dict):
        return '{...}'
    if isinstance(v, (int, np.integer)):
        return f'{int(v)}'
    if isinstance(v, (float, np.floating)):
        return f'{float(v):.4g}'
    return str(v)


def _flatten_stats(stats: dict, prefix: str = '') -> list[tuple[str, str]]:
    """Walk nested stats dict; return [(label, formatted_value), ...]."""
    out: list[tuple[str, str]] = []
    for k, v in stats.items():
        label = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
        if isinstance(v, dict):
            out.extend(_flatten_stats(v, prefix=label))
        else:
            out.append((label, _format_value(v)))
    return out


def write_analysis_table(d: dict, path: str) -> None:
    """Per-analysis markdown — statistics, sample size, summary, notes."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {d.get('name', 'unnamed')}\n")
    lines.append(f"**Sample size:** n = {d.get('n', 0)}\n")
    lines.append(f"**Summary:** {d.get('summary', '')}\n")
    if d.get('notes'):
        lines.append(f"\n**Notes:** {d['notes']}\n")
    lines.append('\n## Statistics\n')
    lines.append('| Name | Value |')
    lines.append('|------|-------|')
    for k, v in _flatten_stats(d.get('statistics') or {}):
        lines.append(f'| `{k}` | {v} |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_summary_md(results: dict[str, dict], path: str) -> None:
    """Top-level summary across all analyses."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    lines = ['# Analysis summary\n']
    lines.append('| Analysis | Sample n |')
    lines.append('|----------|----------|')
    for key, d in results.items():
        lines.append(f'| {key} | {d.get("n", 0)} |')
    lines.append('\n## Per-analysis details')
    for key, d in results.items():
        lines.append(f'\n### {key} — {d.get("name", "")}')
        lines.append(f'\n{d.get("summary", "")}')
        if d.get('notes'):
            lines.append(f'\n*Notes:* {d["notes"]}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ============================================================================
# Figures
# ============================================================================
def write_figures(results: dict[str, dict],
                  df: pd.DataFrame,
                  figures_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[run_analysis] matplotlib not available; skipping figures.')
        return

    os.makedirs(figures_dir, exist_ok=True)
    _fig_variance_decomposition(
        results.get('variance_decomposition', {}),
        os.path.join(figures_dir, 'variance_decomposition.png'), plt,
    )
    _fig_polar_penalty(
        results.get('polar_penalty_contrast', {}), df,
        os.path.join(figures_dir, 'polar_penalty_contrast.png'), plt,
    )
    _fig_characterization_correlations(
        results.get('characterization_correlations', {}),
        os.path.join(figures_dir, 'characterization_correlations.png'), plt,
    )
    _fig_sh_lmax_ablation(
        results.get('sh_lmax_ablation', {}),
        os.path.join(figures_dir, 'sh_lmax_ablation.png'), plt,
    )


def _fig_variance_decomposition(d: dict, path: str, plt) -> None:
    eta = d.get('data', {}).get('eta_sq', {})
    ci  = d.get('data', {}).get('ci', {})
    if not eta:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    factors = list(eta.keys())
    values = [eta[k] for k in factors]
    err_lo = [eta[k] - ci.get(k, [eta[k], eta[k]])[0] for k in factors]
    err_hi = [ci.get(k, [eta[k], eta[k]])[1] - eta[k] for k in factors]
    ax.bar(factors, values, yerr=[err_lo, err_hi], capsize=4,
           color=['#4c72b0', '#dd8452', '#55a868'])
    ax.set_ylabel('η²  (SS_factor / SS_total)')
    ax.set_title('Variance decomposition of held-out PSNR')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_polar_penalty(d: dict, df: pd.DataFrame, path: str, plt) -> None:
    from analysis.decisions import _pe_cell_key
    if not d or 'held_out_psnr_polar' not in df.columns:
        return
    sub = df[df['status'] == 'completed'].copy() if 'status' in df.columns else df
    if sub.empty:
        return
    sub['pe_cell'] = [_pe_cell_key(c, p) for c, p in zip(sub['ce'], sub['pe'])]
    grouped = sub.groupby('pe_cell').agg(
        polar=('held_out_psnr_polar', 'mean'),
        equatorial=('held_out_psnr_equatorial', 'mean'),
    )
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(grouped))
    width = 0.35
    ax.bar(x - width / 2, grouped['polar'], width, label='polar (|φ|>60°)')
    ax.bar(x + width / 2, grouped['equatorial'], width, label='equatorial (|φ|<30°)')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=20, ha='right')
    ax.set_ylabel('held_out PSNR (dB)')
    ax.set_title('Polar vs equatorial PSNR per PE cell')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_characterization_correlations(d: dict, path: str, plt) -> None:
    per_pe = d.get('data', {}).get('per_pe_cell', {})
    if not per_pe:
        return
    pe_cells = list(per_pe.keys())
    if not pe_cells:
        return
    features = list(per_pe[pe_cells[0]].keys())
    mat = np.zeros((len(pe_cells), len(features)), dtype=float)
    for i, pe in enumerate(pe_cells):
        for j, feat in enumerate(features):
            mat[i, j] = per_pe[pe][feat]['rho']
    fig, ax = plt.subplots(figsize=(1.4 * len(features) + 2, 0.5 * len(pe_cells) + 1.5))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=20, ha='right')
    ax.set_yticks(range(len(pe_cells))); ax.set_yticklabels(pe_cells)
    for i in range(len(pe_cells)):
        for j in range(len(features)):
            ax.text(j, i, f'{mat[i, j]:+.2f}', ha='center', va='center',
                    color='black' if abs(mat[i, j]) < 0.6 else 'white', fontsize=9)
    fig.colorbar(im, ax=ax, label='Spearman ρ')
    ax.set_title('Spearman ρ(per-dataset mean PSNR, feature) — n=5 datasets')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_sh_lmax_ablation(d: dict, path: str, plt) -> None:
    post = d.get('data', {}).get('post_saturation_deltas', [])
    pre  = d.get('data', {}).get('pre_saturation_deltas', [])
    if not post and not pre:
        return
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4))
    if post:
        df_p = pd.DataFrame(post)
        for ds, sub in df_p.groupby('dataset'):
            ax_a.scatter([ds] * len(sub), sub['delta_db'], s=40,
                         label=f"{ds} (L_max={int(sub['lmax_matched'].iloc[0])})")
        ax_a.axhline(0, color='k', lw=0.5)
        ax_a.set_ylabel('Δ_match (matched − default L_max=32, dB)')
        ax_a.set_title('Post-saturation (low-bandwidth datasets)')
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)
    if pre:
        df_b = pd.DataFrame(pre)
        for ds, sub in df_b.groupby('dataset'):
            ax_b.scatter([ds] * len(sub), sub['delta_db'], s=40, label=ds)
        ax_b.axhline(0, color='k', lw=0.5)
        ax_b.set_ylabel('Δ_LMax (L_max=32 − L_max=16, dB)')
        ax_b.set_title('Pre-saturation (high-bandwidth datasets)')
        ax_b.legend(fontsize=8)
        ax_b.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--runs_csv', default='results/runs.csv')
    parser.add_argument('--output_dir', default='results/analysis')
    parser.add_argument('--skip_figures', action='store_true',
                        help='Skip figure generation (no matplotlib needed).')
    args = parser.parse_args()

    if not os.path.exists(args.runs_csv):
        print(f"[run_analysis] runs.csv not found at {args.runs_csv}. "
              "Run scripts/run_grid.py first.")
        sys.exit(1)

    df = pd.read_csv(args.runs_csv)
    # pandas' default na_values list includes the string 'None', so a CSV row
    # with `pe=None` (a legitimate value meaning "no positional encoding")
    # silently round-trips to NaN. Restore the literal here.
    for col in ('pe', 'ce', 'act'):
        if col in df.columns:
            df[col] = df[col].fillna('None' if col == 'pe' else '')
    print(f"[run_analysis] Loaded {len(df)} rows from {args.runs_csv}")
    if 'status' in df.columns:
        n_complete = int((df['status'] == 'completed').sum())
        print(f"               {n_complete} rows with status='completed'")

    results: dict[str, dict] = {}
    for key, fn in ANALYSES:
        results[key] = fn(df)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    write_summary_json(results, os.path.join(output_dir, 'summary.json'))
    print(f"[run_analysis] summary.json → {output_dir}/summary.json")

    for key, d in results.items():
        write_analysis_table(d, os.path.join(tables_dir, f'{key}.md'))
    print(f"[run_analysis] tables → {tables_dir}/")

    write_summary_md(results, os.path.join(output_dir, 'summary.md'))
    print(f"[run_analysis] summary → {output_dir}/summary.md")

    if not args.skip_figures:
        write_figures(results, df, figures_dir)
        print(f"[run_analysis] figures → {figures_dir}/")

    # ---- Console summary ----
    print('\n' + '=' * 72)
    print('Analyses')
    print('=' * 72)
    for key, d in results.items():
        n = d.get('n', 0)
        print(f"  {key:<35} [n={n:>3}]  {d.get('summary', '')[:140]}")
    print('=' * 72)


if __name__ == '__main__':
    main()

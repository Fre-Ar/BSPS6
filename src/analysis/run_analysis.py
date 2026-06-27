"""
Main analysis script — consumes results/runs.csv and emits per-hypothesis
decisions, markdown tables, and figures.

Usage from repo root:
    PYTHONPATH=src python src/analysis/run_analysis.py \
        --runs_csv results/runs.csv \
        --output_dir results/analysis

Outputs:
    <output_dir>/decisions.json
    <output_dir>/tables/{h1,h3,h4,h5a,h5b}.md
    <output_dir>/figures/{h1,h3,h4,h5}.png  (unless --skip_figures)
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
    evaluate_h1, evaluate_h3, evaluate_h4,
    evaluate_h5a, evaluate_h5b,
)
from analysis.characterization import DATASET_METRICS                     # noqa: E402


# Hypotheses we apply Holm-Bonferroni across (preregistration §4). H3 and H4
# use threshold-based decisions, not single p-values, so they're not part of
# the multiple-comparison family.
HOLM_FAMILY = ('H1', 'H5a', 'H5b')
ALPHA = 0.05


# ============================================================================
# Holm-Bonferroni
# ============================================================================
def apply_holm_bonferroni(
    decisions: dict[str, dict],
    family: tuple[str, ...] = HOLM_FAMILY,
    alpha: float = ALPHA,
) -> None:
    """In-place Holm-Bonferroni correction across `family`. A hypothesis that
    was 'accepted' before correction may be downgraded to 'inconclusive' if
    its sorted-rank-adjusted p-value exceeds alpha.

    Hypotheses not in `family`, or those without a usable p-value, are left
    unchanged.
    """
    ps: list[tuple[str, float]] = []
    for h in family:
        d = decisions.get(h)
        if d is None:
            continue
        p = d.get('statistic', {}).get('p_value')
        if p is None or not isinstance(p, (int, float)) or not math.isfinite(p):
            continue
        ps.append((h, float(p)))

    # Strict Holm: sort ascending by p; the i-th smallest is significant iff
    # p_(i) ≤ alpha / (m − i + 1) AND every smaller-rank hypothesis was also
    # significant. Once any hypothesis fails, all later ones are non-significant.
    ps.sort(key=lambda x: x[1])
    m = len(ps)
    first_failure_seen = False
    for i, (h, p) in enumerate(ps, start=1):
        adj_threshold = alpha / (m - i + 1)
        d = decisions[h]
        d.setdefault('statistic', {})['holm_threshold'] = adj_threshold
        # In strict Holm the test stops at the first failure; everything
        # at or after a failure is non-significant.
        if first_failure_seen or p > adj_threshold:
            d['statistic']['holm_significant'] = False
            if not first_failure_seen and p > adj_threshold:
                first_failure_seen = True
            # Downgrade an 'accepted' decision that no longer passes Holm.
            if d.get('decision') == 'accepted':
                d['decision_before_holm'] = 'accepted'
                d['decision'] = 'inconclusive'
                d['notes'] = (
                    (d.get('notes', '') + '; ' if d.get('notes') else '')
                    + (f'downgraded to inconclusive after Holm-Bonferroni '
                       f'(p={p:.4f} > {adj_threshold:.4f})'
                       if not first_failure_seen
                       else 'downgraded to inconclusive after Holm-Bonferroni '
                            '(smaller-p hypothesis already failed)')
                ).lstrip('; ')
        else:
            d['statistic']['holm_significant'] = True


# ============================================================================
# Output writers — JSON
# ============================================================================
def _json_safe(obj):
    """Convert numpy / tuple types to plain Python for JSON serialization."""
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


def write_decisions_json(decisions: dict[str, dict], path: str) -> None:
    """One JSON file aggregating every hypothesis decision."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_json_safe(decisions), f, indent=2, sort_keys=False)


# ============================================================================
# Output writers — markdown tables
# ============================================================================
def write_decision_table(d: dict, path: str) -> None:
    """Per-hypothesis markdown summarising stat / threshold / observed / decision."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {d.get('name', 'unnamed')}\n")
    lines.append(f"**Decision:** `{d.get('decision', 'unknown')}`\n")
    lines.append(f"**Sample size:** n = {d.get('n', 0)}\n")
    lines.append(f"**Reasoning:** {d.get('reasoning', '')}\n")
    if d.get('notes'):
        lines.append(f"**Notes:** {d['notes']}\n")
    lines.append('\n## Statistics\n')
    lines.append('| Name | Value |')
    lines.append('|------|-------|')
    for k, v in (d.get('statistic') or {}).items():
        lines.append(f'| `{k}` | {_format_value(v)} |')
    lines.append('\n## Pre-committed thresholds\n')
    lines.append('| Name | Value |')
    lines.append('|------|-------|')
    for k, v in (d.get('threshold') or {}).items():
        lines.append(f'| `{k}` | {_format_value(v)} |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _format_value(v) -> str:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            lo, hi = float(v[0]), float(v[1])
            return f'[{lo:.3f}, {hi:.3f}]'
        except (TypeError, ValueError):
            pass
    if isinstance(v, (dict,)):
        return '{...}'
    if isinstance(v, (int, np.integer)):
        return f'{int(v)}'
    if isinstance(v, (float, np.floating)):
        return f'{float(v):.4g}'
    return str(v)


def write_summary_md(decisions: dict[str, dict], path: str) -> None:
    """Top-level summary table covering every hypothesis."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    lines = ['# Hypothesis decisions — summary\n']
    lines.append('| Hypothesis | Decision | Pre-Holm | Sample n |')
    lines.append('|------------|----------|----------|----------|')
    for h, d in decisions.items():
        decision = d.get('decision', 'unknown')
        pre_holm = d.get('decision_before_holm', '—')
        n = d.get('n', 0)
        lines.append(f'| {h} | `{decision}` | `{pre_holm}` | {n} |')
    lines.append('\n## Per-hypothesis details')
    for h, d in decisions.items():
        lines.append(f'\n### {h} — {d.get("name", "")}')
        lines.append(f'**Decision:** `{d.get("decision", "unknown")}`')
        lines.append(f'\n{d.get("reasoning", "")}')
        if d.get('notes'):
            lines.append(f'\n*Notes:* {d["notes"]}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ============================================================================
# Output writers — figures
# ============================================================================
def write_figures(decisions: dict[str, dict],
                  df: pd.DataFrame,
                  figures_dir: str) -> None:
    """Emit the 5 figures from preregistration §6. matplotlib is imported
    lazily so the analysis CLI can run without it for --skip_figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')                              # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print('[run_analysis] matplotlib not available; skipping figures.')
        return

    os.makedirs(figures_dir, exist_ok=True)
    _fig_h1(decisions.get('H1', {}), df, os.path.join(figures_dir, 'h1.png'), plt)
    _fig_h3(decisions.get('H3', {}), os.path.join(figures_dir, 'h3.png'), plt)
    _fig_h4(decisions.get('H4', {}), os.path.join(figures_dir, 'h4.png'), plt)
    _fig_h5_lmax(decisions.get('H5a', {}), decisions.get('H5b', {}),
                 os.path.join(figures_dir, 'h5.png'), plt)


def _fig_h1(d: dict, df: pd.DataFrame, path: str, plt) -> None:
    """Bar chart: mean polar PSNR vs mean equatorial PSNR per PE cell."""
    from analysis.decisions import _pe_cell_key
    if not d or d.get('decision', '') == 'unknown' or 'held_out_psnr_polar' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df['status'] == 'completed'].copy() if 'status' in df.columns else df
    if sub.empty:
        plt.close(fig); return
    sub['pe_cell'] = [_pe_cell_key(c, p) for c, p in zip(sub['ce'], sub['pe'])]
    grouped = sub.groupby('pe_cell').agg(
        polar=('held_out_psnr_polar', 'mean'),
        equatorial=('held_out_psnr_equatorial', 'mean'),
    )
    if grouped.empty:
        plt.close(fig); return
    x = np.arange(len(grouped))
    width = 0.35
    ax.bar(x - width / 2, grouped['polar'], width, label='polar (|φ|>60°)')
    ax.bar(x + width / 2, grouped['equatorial'], width, label='equatorial (|φ|<30°)')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=20, ha='right')
    ax.set_ylabel('held_out PSNR (dB)')
    ax.set_title(f"H1 — Polar vs equatorial PSNR per PE cell "
                 f"(decision: {d.get('decision', '?')})")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_h3(d: dict, path: str, plt) -> None:
    """Bar chart of partial η² with 95% bootstrap CIs."""
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
    ax.set_ylabel('partial η²')
    ax.set_title(f"H3 — Variance decomposition (decision: {d.get('decision', '?')})")
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_h4(d: dict, path: str, plt) -> None:
    """Scatter of LOO-CV predicted vs observed PSNR, one subplot per PE cell."""
    per_pe = d.get('data', {}).get('per_pe_cell', {})
    if not per_pe:
        return
    pe_cells = [k for k, v in per_pe.items() if v.get('y_true')]
    if not pe_cells:
        return
    n = len(pe_cells)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), squeeze=False)
    for ax, pe in zip(axes[0], pe_cells):
        info = per_pe[pe]
        yt = np.asarray(info['y_true']); yp = np.asarray(info['y_pred'])
        ax.scatter(yp, yt, s=14, alpha=0.7)
        if yt.size > 0:
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, lw=1)
        ax.set_xlabel('predicted PSNR (LOO-CV)')
        ax.set_ylabel('observed PSNR')
        ci_lo = info.get('ci_lo'); ci_hi = info.get('ci_hi')
        ci_txt = (f"CI [{ci_lo:.2f},{ci_hi:.2f}]"
                  if isinstance(ci_lo, float) and isinstance(ci_hi, float)
                  and math.isfinite(ci_lo) and math.isfinite(ci_hi)
                  else "CI n/a")
        ax.set_title(f"{pe}\nR²={info['r2']:.3f}, {ci_txt}", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"H4 — Characterization predicts PSNR per PE cell "
                 f"(decision: {d.get('decision', '?')})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_h5_lmax(d_h5a: dict, d_h5b: dict, path: str, plt) -> None:
    """Combined H5a/H5b: per-dataset deltas across architectures."""
    deltas_a = d_h5a.get('data', {}).get('deltas', [])
    deltas_b = d_h5b.get('data', {}).get('deltas', [])
    if not deltas_a and not deltas_b:
        return
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4))

    if deltas_a:
        df_a = pd.DataFrame(deltas_a)
        # Earlier schema used 'arch' as the within-dataset key; the
        # redesigned schema uses 'activation'. Accept either.
        for ds, sub in df_a.groupby('dataset'):
            ax_a.scatter([ds] * len(sub), sub['delta_db'],
                         s=40, label=f"{ds} (L_max={int(sub['lmax_matched'].iloc[0])})")
        ax_a.axhline(0, color='k', lw=0.5)
        ax_a.axhline(-0.5, color='r', lw=0.5, linestyle='--', label='accept threshold (−0.5 dB)')
        ax_a.set_ylabel('Δ_match (matched − default L_max=32, dB)')
        ax_a.set_title(f"H5a — post-saturation (decision: {d_h5a.get('decision', '?')})")
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)

    if deltas_b:
        df_b = pd.DataFrame(deltas_b)
        for ds, sub in df_b.groupby('dataset'):
            ax_b.scatter([ds] * len(sub), sub['delta_db'], s=40, label=ds)
        ax_b.axhline(0, color='k', lw=0.5)
        ax_b.axhline(0.5, color='g', lw=0.5, linestyle='--', label='accept threshold (+0.5 dB)')
        ax_b.set_ylabel('Δ_LMax (L_max=32 − L_max=16, dB)')
        ax_b.set_title(f"H5b — pre-saturation (decision: {d_h5b.get('decision', '?')})")
        ax_b.legend(fontsize=8)
        ax_b.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs_csv', default='results/runs.csv')
    parser.add_argument('--output_dir', default='results/analysis')
    parser.add_argument('--skip_figures', action='store_true',
                        help='Skip figure generation (faster, no matplotlib needed).')
    parser.add_argument('--skip_holm', action='store_true',
                        help='Skip Holm-Bonferroni correction across H1/H5a/H5b.')
    args = parser.parse_args()

    if not os.path.exists(args.runs_csv):
        print(f"[run_analysis] runs.csv not found at {args.runs_csv}. "
              "Run scripts/run_grid.py first.")
        sys.exit(1)

    df = pd.read_csv(args.runs_csv)
    # pandas' default `na_values` list includes the string 'None', so a CSV
    # row with `pe=None` (a legitimate value meaning "no positional encoding")
    # silently round-trips to NaN. That cascades into _pe_cell_key returning
    # bogus '<ce>_nan' labels and H1/H3/H4 then losing all the
    # `pe in {none_angular, none_cartesian, sh}` rows. Restore the literal.
    for col in ('pe', 'ce', 'arch', 'act'):
        if col in df.columns:
            df[col] = df[col].fillna('None' if col == 'pe' else '')
            
    print(f"[run_analysis] Loaded {len(df)} rows from {args.runs_csv}")
    if 'status' in df.columns:
        n_complete = int((df['status'] == 'completed').sum())
        print(f"               {n_complete} rows with status='completed'")

    decisions: dict[str, dict] = {
        'H1':   evaluate_h1(df),
        'H3':   evaluate_h3(df),
        'H4':   evaluate_h4(df),
        'H5a':  evaluate_h5a(df),
        'H5b':  evaluate_h5b(df),
    }

    if not args.skip_holm:
        apply_holm_bonferroni(decisions)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    write_decisions_json(decisions, os.path.join(output_dir, 'decisions.json'))
    print(f"[run_analysis] decisions.json → {output_dir}/decisions.json")

    for h, d in decisions.items():
        fname = f"{h.lower().replace('.', '_').replace(' ', '_')}.md"
        write_decision_table(d, os.path.join(tables_dir, fname))
    print(f"[run_analysis] tables → {tables_dir}/")

    write_summary_md(decisions, os.path.join(output_dir, 'summary.md'))
    print(f"[run_analysis] summary → {output_dir}/summary.md")

    if not args.skip_figures:
        write_figures(decisions, df, figures_dir)
        print(f"[run_analysis] figures → {figures_dir}/")

    # ---- Console summary ----
    print('\n' + '=' * 72)
    print('Hypothesis decisions')
    print('=' * 72)
    for h, d in decisions.items():
        decision = d.get('decision', 'unknown')
        n = d.get('n', 0)
        print(f"  {h:<6} [n={n:>3}]  {decision:<14}  {d.get('reasoning', '')[:90]}")
    print('=' * 72)


if __name__ == '__main__':
    main()

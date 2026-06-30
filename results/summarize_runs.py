"""Condense results/runs.csv into a per-run summary CSV.

Drops the constant hyperparameter / timestamp / log-dir columns, rounds
floats to 2 decimals, and replaces (ce, pe, act) with a short cell key
like `ssine+sh`. RGB-only columns stay blank for scalar datasets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math


# (ce, pe) → PE-cell short code; activation → short code.
_PE_SHORT = {
    ('angular',             'None'): 'ang',
    ('cartesian',           'None'): 'cart',
    ('cartesian',           'RFF'):  'rff',
    ('spherical-harmonics', 'None'): 'sh',
    ('cartesian',           'FKAN'): 'fkan',
}
_ACT_SHORT = {'relu': 'relu', 'scaled-sine': 'ssine', 'gaussian': 'gauss'}


def _f(s: str) -> str:
    """Two-decimal float; blank for empty / NaN."""
    if not s:
        return ''
    try:
        v = float(s)
    except ValueError:
        return s
    return '' if math.isnan(v) else f'{v:.2f}'


def _summarize(row: dict) -> dict:
    cell = f"{_ACT_SHORT.get(row['act'], row['act'])}+" \
           f"{_PE_SHORT.get((row['ce'], row['pe']), row['pe'])}"
    lmax = ''
    if row['ce'] == 'spherical-harmonics':
        try:
            lmax = str(json.loads(row.get('encoding_kwargs_json') or '{}')
                       .get('L_max', ''))
        except json.JSONDecodeError:
            pass
    return {
        'cell':    cell,
        'dataset': row['dataset'],
        'lmax':    lmax,
        'recon_o': _f(row['reconstruction_psnr']),
        'recon_p': _f(row['reconstruction_psnr_polar']),
        'recon_e': _f(row['reconstruction_psnr_equatorial']),
        'held_o':  _f(row['held_out_psnr']),
        'held_p':  _f(row['held_out_psnr_polar']),
        'held_e':  _f(row['held_out_psnr_equatorial']),
        'recon_R': _f(row['reconstruction_psnr_r']),
        'recon_G': _f(row['reconstruction_psnr_g']),
        'recon_B': _f(row['reconstruction_psnr_b']),
        'held_R':  _f(row['held_out_psnr_r']),
        'held_G':  _f(row['held_out_psnr_g']),
        'held_B':  _f(row['held_out_psnr_b']),
        'epochs':  row['epochs_run'],
        'secs':    str(int(round(float(row['wall_clock_seconds']))))
                   if row['wall_clock_seconds'] else '',
        'params':  f"{int(float(row['parameter_count'])) // 1000}K"
                   if row['parameter_count'] else '',
    }


def _sort_key(r: dict) -> tuple:
    # cell, dataset, then SH L_max (32 default first, then 16, then post-sat)
    lmax_order = {'': 0, '32': 1, '16': 2, '13': 3, '31': 4}
    return r['cell'], r['dataset'], lmax_order.get(r['lmax'], 9)


def main(runs_csv: str = 'results/runs.csv', out_csv: str = 'results/runs_summary.csv') -> None:

    with open(runs_csv, newline='') as f:
        rows = sorted((_summarize(r) for r in csv.DictReader(f)), key=_sort_key)

    fields = list(rows[0].keys()) if rows else []
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {out_csv}')


if __name__ == '__main__':
    main()

"""
Characterize a pre-processed spherical dataset along the four axes.
"""
from __future__ import annotations

from datasets.characteristics import (
    characterize_spherical_dataset, print_and_plot_results, _print_scalar_summary
)
from config.constants import DATASET_TYPES, DATASET_CHOICES, DATASET_CONFIG


def main(dataset: DATASET_TYPES = 'etopo1', path: str = None, no_plot: bool = False) -> None:
    path = path or DATASET_CONFIG[dataset]['path']
    results = characterize_spherical_dataset(path)

    if no_plot:
        # Print only the summary text
        if results.get('kind') == 'scalar':
            _print_scalar_summary(results, header=f'{dataset.upper()} (scalar)')
        else:
            _print_scalar_summary(results['luminance'],
                                  header=f'{dataset.upper()} LUMINANCE')
            for ch in ('R', 'G', 'B'):
                _print_scalar_summary(results['per_channel'][ch],
                                      header=f'{dataset.upper()} {ch}')
    else:
        print_and_plot_results(results)


if __name__ == '__main__':
    #main('etopo1')
    #main('era5')
    #main('cmb')
    #main('hdri_sky')
    main('hdri_urban')
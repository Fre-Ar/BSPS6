from config.constants import (
    RAW_ETOPO1_PATH, ELEVATION_DATA_PATH,
    RAW_ERA5_PATH,   ERA5_DATA_PATH,
    RAW_CMB_PATH,    CMB_DATA_PATH,
    RAW_HDRI_SKY_PATH,   HDRI_SKY_DATA_PATH,
    RAW_HDRI_URBAN_PATH,   HDRI_URBAN_DATA_PATH,
    DATASET_CHOICES, BENCH_LAT, BENCH_LON
)
from datasets.preprocessing import (
    preprocess_etopo1, preprocess_era5, preprocess_cmb, preprocess_hdri,
)

_DEFAULT_PATHS = {
    'etopo1':     (RAW_ETOPO1_PATH,     ELEVATION_DATA_PATH,  preprocess_etopo1),
    'era5':       (RAW_ERA5_PATH,       ERA5_DATA_PATH,       preprocess_era5),
    'cmb':        (RAW_CMB_PATH,        CMB_DATA_PATH,        preprocess_cmb),
    'hdri_sky':   (RAW_HDRI_SKY_PATH,   HDRI_SKY_DATA_PATH,   preprocess_hdri),
    'hdri_urban': (RAW_HDRI_URBAN_PATH, HDRI_URBAN_DATA_PATH, preprocess_hdri),
}

def _run_one(name: str, src: str | None, dst: str | None) -> None:
    default_src, default_dst, fn = _DEFAULT_PATHS[name]
    fn(src or default_src, dst or default_dst, BENCH_LAT, BENCH_LON)

def main(dataset: str, src: str | None = None, dst: str | None = None):
    assert dataset in DATASET_CHOICES + ('all',), f"Invalid dataset: {dataset!r}"
    
    if dataset == 'all':
        for name in DATASET_CHOICES:
            print(f"\n{'='*60}\n== {name.upper()}\n{'='*60}")
            try:
                _run_one(name, None, None)
            except Exception as e:  
                print(f"[{name}] FAILED: {e!r}")
    else:
        _run_one(dataset, src, dst)


if __name__ == '__main__':
    main('hdri_urban')
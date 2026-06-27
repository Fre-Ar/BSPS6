from config.constants import (
    RAW_ETOPO1_PATH,     ELEVATION_DATA_PATH,  ELEVATION_HELD_OUT_PATH,
    RAW_ERA5_PATH,       ERA5_DATA_PATH,       ERA5_HELD_OUT_PATH,
    RAW_CMB_PATH,        CMB_DATA_PATH,        CMB_HELD_OUT_PATH,
    RAW_HDRI_SKY_PATH,   HDRI_SKY_DATA_PATH,   HDRI_SKY_HELD_OUT_PATH,
    RAW_HDRI_URBAN_PATH, HDRI_URBAN_DATA_PATH, HDRI_URBAN_HELD_OUT_PATH,
    DATASET_CHOICES, BENCH_LAT, BENCH_LON
)
from datasets.preprocessing import (
    preprocess_etopo1, preprocess_era5, preprocess_cmb, preprocess_hdri,
)

# (raw_source_path, training_output_path, held_out_output_path, fn)
_DEFAULT_PATHS = {
    'etopo1':     (RAW_ETOPO1_PATH,     ELEVATION_DATA_PATH,  ELEVATION_HELD_OUT_PATH,  preprocess_etopo1),
    'era5':       (RAW_ERA5_PATH,       ERA5_DATA_PATH,       ERA5_HELD_OUT_PATH,       preprocess_era5),
    'cmb':        (RAW_CMB_PATH,        CMB_DATA_PATH,        CMB_HELD_OUT_PATH,        preprocess_cmb),
    'hdri_sky':   (RAW_HDRI_SKY_PATH,   HDRI_SKY_DATA_PATH,   HDRI_SKY_HELD_OUT_PATH,   preprocess_hdri),
    'hdri_urban': (RAW_HDRI_URBAN_PATH, HDRI_URBAN_DATA_PATH, HDRI_URBAN_HELD_OUT_PATH, preprocess_hdri),
}

def _run_one(name: str, src: str | None, dst: str | None,
             held_out: str | None = None, skip_held_out: bool = False) -> None:
    default_src, default_dst, default_held_out, fn = _DEFAULT_PATHS[name]
    held_out_path = None if skip_held_out else (held_out or default_held_out)
    fn(src or default_src, dst or default_dst,
       held_out_filepath=held_out_path)

def main(dataset: str, src: str | None = None, dst: str | None = None,
         held_out: str | None = None, skip_held_out: bool = False):
    assert dataset in DATASET_CHOICES + ('all',), f"Invalid dataset: {dataset!r}"
    
    if dataset == 'all':
        for name in DATASET_CHOICES:
            print(f"\n{'='*60}\n== {name.upper()}\n{'='*60}")
            try:
                _run_one(name, None, None,
                         held_out=None, skip_held_out=skip_held_out)
            except Exception as e:  
                print(f"[{name}] FAILED: {e!r}")
    else:
        _run_one(dataset, src, dst, held_out, skip_held_out)


if __name__ == '__main__':
    main('all')
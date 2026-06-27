from typing import Literal, get_args

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATASETS_DIR = "src/datasets/files"

# Pre-processed dataset files
ELEVATION_DATA_PATH  = f"{DATASETS_DIR}/ETOPO1_512x1024.nc"
ERA5_DATA_PATH       = f"{DATASETS_DIR}/ERA5_t2m_2023_06_15_1200_512x1024.nc"
CMB_DATA_PATH        = f"{DATASETS_DIR}/CMB_SMICA_Full_512x1024.nc"
HDRI_SKY_DATA_PATH   = f"{DATASETS_DIR}/HDRI_kloofendal_2k_512x1024.nc"
HDRI_URBAN_DATA_PATH = f"{DATASETS_DIR}/HDRI_shanghai_2k_512x1024.nc"

# Held-out evaluation files: 511×1023 half-pixel-offset grid sampled directly
# from each dataset's BASE source (not interpolated from the 512×1024 training
# file). Produced by the same preprocessors when `held_out_filepath` is provided. 
ELEVATION_HELD_OUT_PATH  = f"{DATASETS_DIR}/ETOPO1_511x1023_held_out.nc"
ERA5_HELD_OUT_PATH       = f"{DATASETS_DIR}/ERA5_t2m_2023_06_15_1200_511x1023_held_out.nc"
CMB_HELD_OUT_PATH        = f"{DATASETS_DIR}/CMB_SMICA_Full_511x1023_held_out.nc"
HDRI_SKY_HELD_OUT_PATH   = f"{DATASETS_DIR}/HDRI_kloofendal_2k_511x1023_held_out.nc"
HDRI_URBAN_HELD_OUT_PATH = f"{DATASETS_DIR}/HDRI_shanghai_bund_511x1023_held_out.nc"

# Raw source files 
RAW_ETOPO1_PATH     = f"{DATASETS_DIR}/ETOPO1_Ice_g_gmt4.grd"
RAW_ERA5_PATH       = f"{DATASETS_DIR}/ERA5_t2m_2023_06_15_1200.nc"
RAW_CMB_PATH        = f"{DATASETS_DIR}/COM_CMB_IQU-smica_2048_R3.00_full.fits"
RAW_HDRI_SKY_PATH   = f"{DATASETS_DIR}/kloofendal_48d_partly_cloudy_puresky_2k.exr"
RAW_HDRI_URBAN_PATH = f"{DATASETS_DIR}/shanghai_bund_2k.exr"
 
 
# ---------------------------------------------------------------------
# String Constants
# ---------------------------------------------------------------------
COORD = "coord"
TARGET = "target"

# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------
CE_TYPES = Literal['angular', 'cartesian', 'spherical-harmonics', 'spherical-rff']
CE_CHOICES: tuple[CE_TYPES, ...] = get_args(CE_TYPES)
 
DATASET_TYPES = Literal['etopo1', 'era5', 'cmb', 'hdri_sky', 'hdri_urban']
DATASET_CHOICES: tuple[DATASET_TYPES, ...] = get_args(DATASET_TYPES)
 
# Config per dataset: (pre-processed path, number of output channels).
DATASET_CONFIG: dict[str, dict] = {
    'etopo1':     {'path': ELEVATION_DATA_PATH,  'held_out_path': ELEVATION_HELD_OUT_PATH,  'out_features': 1},
    'era5':       {'path': ERA5_DATA_PATH,       'held_out_path': ERA5_HELD_OUT_PATH,       'out_features': 1},
    'cmb':        {'path': CMB_DATA_PATH,        'held_out_path': CMB_HELD_OUT_PATH,        'out_features': 1},
    'hdri_sky':   {'path': HDRI_SKY_DATA_PATH,   'held_out_path': HDRI_SKY_HELD_OUT_PATH,   'out_features': 3},
    'hdri_urban': {'path': HDRI_URBAN_DATA_PATH, 'held_out_path': HDRI_URBAN_HELD_OUT_PATH, 'out_features': 3},
}

 
# Standard benchmark grid resolution (lat, lon). 
BENCH_LAT = 512
BENCH_LON = 1024
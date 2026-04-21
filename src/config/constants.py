from typing import Literal, get_args

# Paths
ELEVATION_DATA_PATH = "src/datasets/files/ETOPO1_512x1024.nc"

# String Constants
COORD = "coord"
TARGET = "target"

# Enums
CE_TYPES = Literal['angular', 'cartesian', 'spherical-harmonics', 'spherical-rff']
CE_CHOICES: tuple[CE_TYPES, ...] = get_args(CE_TYPES)


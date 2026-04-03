import xarray as xr
import numpy as np

def downsample_etopo1(input_filepath: str, output_filepath: str):
    print("Loading original ETOPO1 dataset...")
    # Open the original high-resolution ETOPO1 dataset
    ds = xr.open_dataset(input_filepath)
    
    # ETOPO1 usually maps longitude to 'x' and latitude to 'y'
    # Check your specific file; if it uses 'lon' and 'lat', replace 'x' and 'y' below.
    lat_name = 'y' if 'y' in ds.dims else 'lat'
    lon_name = 'x' if 'x' in ds.dims else 'lon'

    # 1. Define the new target resolution (e.g., 512 latitude x 1024 longitude)
    # This yields exactly 524,288 points.
    target_lat_size = 512
    target_lon_size = 1024

    print(f"Creating new grid: {target_lat_size}x{target_lon_size}...")
    # 2. Create new coordinate arrays spanning the exact geographic bounds of the original
    new_lats = np.linspace(ds[lat_name].min().values, ds[lat_name].max().values, target_lat_size)
    new_lons = np.linspace(ds[lon_name].min().values, ds[lon_name].max().values, target_lon_size)

    # 3. Interpolate the dataset to the new, coarser grid
    # 'linear' interpolation is standard for continuous topography data
    print("Interpolating data (this may take a minute due to the 233M points)...")
    ds_downsampled = ds.interp({lat_name: new_lats, lon_name: new_lons}, method='linear')

    # 4. Save to NetCDF format
    print(f"Saving downsampled data to {output_filepath}...")
    ds_downsampled.to_netcdf(output_filepath)
    print("Done!")

if __name__ == "__main__":
    # Replace these paths with your actual local file paths
    IN_FILE = "src/datasets/files/ETOPO1_Ice_g_gmt4.grd" 
    OUT_FILE = "src/datasets/files/ETOPO1_512x1024.nc"
    
    downsample_etopo1(IN_FILE, OUT_FILE)
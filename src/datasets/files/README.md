Place the following datasets in this directory:
- [ETOPO1_Ice_g_gmt4.grd](https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/)
- [ERA5_t2m_2023_06_15_1200.nc](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- [kloofendal_48d_partly_cloudy_puresky_2k.exr](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky)
- [COM_CMB_IQU-smica_2048_R3.00_full.fits](https://pla.esac.esa.int/#maps)

Then run `src/datasets/pre_process.py` to get the following files:
- ETOPO1_512x1024.nc
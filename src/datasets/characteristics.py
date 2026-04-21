import xarray as xr
import numpy as np
import pyshtools as pysh
import matplotlib.pyplot as plt

def characterize_spherical_dataset(filepath: str, var_name='z', lat_name='y', lon_name='x'):
    print(f"Loading dataset: {filepath}...")
    ds = xr.open_dataset(filepath)
    
    # Extract the 2D data array and 1D coordinate arrays
    data = ds[var_name].values
    lats = ds[lat_name].values if lat_name in ds else ds['lat'].values
    lons = ds[lon_name].values if lon_name in ds else ds['lon'].values
    
    print("Computing characteristics...")
    results = {}

    # ---------------------------------------------------------
    # A & B: Spectral Complexity and Effective Bandwidth
    # ---------------------------------------------------------
    # Create a Spherical Harmonic Grid object.
    # 'DH' (Driscoll and Healy) is the standard type for equally spaced lat/lon grids.
    grid = pysh.SHGrid.from_array(data, grid='DH')
    
    # Expand the spatial grid into Spherical Harmonic coefficients
    coeffs = grid.expand()
    
    # Get the power per degree l (C_l)
    power_spectrum = coeffs.spectrum()
    degrees = coeffs.degrees()
    
    # Calculate Effective Bandwidth (L_95%)
    total_power = np.sum(power_spectrum)
    cumulative_power = np.cumsum(power_spectrum)
    l_95 = np.argmax(cumulative_power >= 0.95 * total_power)
    
    results['degrees'] = degrees
    results['power_spectrum'] = power_spectrum
    results['L_95'] = l_95
    results['total_power'] = total_power

    # ---------------------------------------------------------
    # C: Isotropy (Latitudinal Variance)
    # ---------------------------------------------------------
    # Calculate the variance of the signal along each latitude band
    # axis=1 computes variance across the longitudes (x-axis)
    lat_variances = np.var(data, axis=1)
    results['lats'] = lats
    results['lat_variances'] = lat_variances
    
    # For a single summary metric, we can use the Coefficient of Variation of the variances.
    # High CV means highly anisotropic (variance changes wildly depending on latitude).
    results['isotropy_cv'] = np.std(lat_variances) / np.mean(lat_variances)

    # ---------------------------------------------------------
    # D: Dynamic Range and Spatial Gradient
    # ---------------------------------------------------------
    results['dynamic_range'] = np.max(data) - np.min(data)
    results['min_val'] = np.min(data)
    results['max_val'] = np.max(data)
    
    # Calculate 2D gradients (dy, dx)
    dy, dx = np.gradient(data)
    
    # Correct the longitudinal gradient (dx) for converging meridians
    # lats are in degrees, convert to radians
    lats_rad = np.radians(lats)
    cos_lats = np.cos(lats_rad)
    # Clip to avoid division by zero exactly at the poles (-90, 90)
    cos_lats = np.clip(cos_lats, 1e-5, 1.0) 
    
    # Broadcast cos_lats to match 2D data shape
    cos_lats_2d = cos_lats[:, np.newaxis]
    
    # Scale longitudinal gradient 
    dx_scaled = dx / cos_lats_2d
    
    # Calculate gradient magnitude vector
    grad_magnitude = np.sqrt(dx_scaled**2 + dy**2)
    
    results['mean_gradient'] = np.mean(grad_magnitude)
    results['p99_gradient'] = np.percentile(grad_magnitude, 99)

    return results

def print_and_plot_results(results):
    # Print the summary table
    print("\n" + "="*50)
    print(" DATASET CHARACTERISTICS SUMMARY ")
    print("="*50)
    print(f"Dynamic Range:      {results['dynamic_range']:.2f} (Min: {results['min_val']:.2f}, Max: {results['max_val']:.2f})")
    print(f"Mean Gradient:      {results['mean_gradient']:.2f} per pixel")
    print(f"99th %ile Gradient: {results['p99_gradient']:.2f} per pixel (Sharpness)")
    print("-" * 50)
    print(f"Isotropy CV:        {results['isotropy_cv']:.4f} (Higher = more anisotropic)")
    print("-" * 50)
    print(f"Effective Bandwidth (L_95%): Degree {results['L_95']}")
    print("="*50 + "\n")

    # Plot 1: Power Spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results['degrees'], results['power_spectrum'], color='blue', label='Power (C_l)')
    ax1.axvline(x=results['L_95'], color='red', linestyle='--', label=f'L_95% = {results["L_95"]}')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title("Spherical Harmonic Power Spectrum")
    ax1.set_xlabel("Degree (l)")
    ax1.set_ylabel("Power (C_l)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # Plot 2: Latitudinal Variance (Isotropy)
    # We plot latitude on the Y-axis to visually match a map
    ax2.plot(results['lat_variances'], results['lats'], color='green')
    ax2.set_title("Latitudinal Variance (Anisotropy Check)")
    ax2.set_xlabel("Variance across Longitudes")
    ax2.set_ylabel("Latitude (Degrees)")
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

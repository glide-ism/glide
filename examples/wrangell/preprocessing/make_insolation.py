"""
Compute monthly solar potential for Wrangell ice cap.

Uses GLARE's SolarPotential class to compute terrain-corrected insolation
accounting for slope, aspect, and self-shadowing from the DEM.
Saves results to NetCDF in the style of make_pancarra_vars.py.
"""

import calendar
import datetime

from glide.backend import xp as cp
import numpy as np
import xarray as xr

from glare import SolarPotential

# Configuration
VISUALIZE = True
GEOM_PATH = 'model_inputs/gridded_dem.nc'
OUTPUT_PATH = 'model_inputs/gridded_insolation.nc'
GRID_RESOLUTION = 90.0  # meters
LATITUDE = 61.0
LONGITUDE = -143.0
TIMEZONE = "America/Anchorage"
YEAR = 2011

# Load DEM
print(f"Loading DEM from {GEOM_PATH}")
dem = xr.load_dataset(GEOM_PATH)
print(f"DEM shape: {dem.elevation.shape}")

# Initialize solar potential calculator
print("Initializing SolarPotential calculator...")
solar = SolarPotential(
    dem=dem,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    grid_resolution=GRID_RESOLUTION,
    timezone=TIMEZONE,
)

solar_potential_mean, solar_potential_cos, solar_potential_sin = solar.compute_solar_potential_fourier_decomposition(YEAR)

solar_potential_mean_da = xr.DataArray(
    solar_potential_mean.get(),
    dims=["t","y", "x"],
    coords={
        "t": np.arange(0, 12, dtype=np.float32)/12,
        "y": dem.y,
        "x": dem.x,
    },
    attrs={
        "units": "dimensionless ( intensity relative to continuous orthogonal sunlight)",
        "long_name": "Monthly average daily solar potential (incidence-weighted, shadow-masked)",
    }
)

solar_potential_cos_da = xr.DataArray(
    solar_potential_cos.get(),
    dims=["t","y", "x"],
    coords={
        "t": np.arange(0, 12, dtype=np.float32)/12,
        "y": dem.y,
        "x": dem.x,
    },
    attrs={
        "units": "dimensionless ( intensity relative to continuous orthogonal sunlight)",
        "long_name": "cos mode of diurnal variability in insolation",
    }
)
solar_potential_sin_da = xr.DataArray(
    solar_potential_sin.get(),
    dims=["t","y", "x"],
    coords={
        "t": np.arange(0, 12, dtype=np.float32)/12,
        "y": dem.y,
        "x": dem.x,
    },
    attrs={
        "units": "dimensionless ( intensity relative to continuous orthogonal sunlight)",
        "long_name": "sin mode of diurnal variability in insolation",
    }
)

# Create output Dataset by copying DEM and adding new variables
out_ds = dem.copy()
out_ds["monthly_solar_potential_mean"] = solar_potential_mean_da
out_ds["monthly_solar_potential_cos"] = solar_potential_cos_da
out_ds["monthly_solar_potential_sin"] = solar_potential_sin_da

# Add global attributes
out_ds.attrs["source"] = f"GLARE SolarPotential calculator, {YEAR}"
out_ds.attrs["location"] = f"Wrangell Ice Cap (lat={LATITUDE}, lon={LONGITUDE})"
out_ds.attrs["grid_resolution"] = f"{GRID_RESOLUTION} m"

del out_ds['elevation']
del out_ds['domain_mask']
del out_ds['rgi_mask']

# Save to NetCDF
print(f"\nSaving to {OUTPUT_PATH}...")
out_ds.to_netcdf(OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH}")

if VISUALIZE:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    # Visualization
    print("\nCreating visualization...")
    z_cpu = solar.z.get()
    ls = LightSource(azdeg=315, altdeg=45)
    dx = dem.x[1].item() - dem.x[0].item()  # Convert to float
    hs = ls.hillshade(z_cpu, vert_exag=3, dx=dx, dy=dx)

    # Plot April (month 3, 0-indexed) for visualization
    april_idx = 3

    # Solar potential overlay
    plt.imshow(hs, cmap=plt.cm.gray)
    im1 = plt.imshow(solar_potential_mean[april_idx].get(), alpha=0.5, cmap=plt.cm.plasma)
    plt.title('April: Daily Average Solar Potential')
    plt.colorbar(im1, label='Incidence-weighted (dimensionless)')

    plt.tight_layout()
    plt.show()

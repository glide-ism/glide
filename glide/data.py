"""
Data download and loading utilities.

Provides functions to download standard ice sheet datasets and load them
into GLIDE-compatible format.
"""

import os
import urllib.request
from pathlib import Path
import numpy as np

# Default cache directory
CACHE_DIR = Path.home() / ".cache" / "glide"


def get_cache_dir():
    """Get or create the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def download_file(url, filename, cache_dir=None, force=False):
    """
    Download a file with caching.

    Parameters
    ----------
    url : str
        URL to download from
    filename : str
        Local filename to save as
    cache_dir : Path, optional
        Cache directory (default ~/.cache/glide)
    force : bool
        Re-download even if file exists

    Returns
    -------
    Path
        Path to downloaded file
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    filepath = cache_dir / filename

    if filepath.exists() and not force:
        print(f"Using cached: {filepath}")
        return filepath

    print(f"Downloading {url}...")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r  Progress: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
    print()

    return filepath


def load_bedmachine_greenland(path, skip=1, thklim=0.1):
    """
    Load BedMachine Greenland data.

    Parameters
    ----------
    path : str or Path
        Path to BedMachineGreenland-v5.nc
    skip : int
        Downsampling factor (default 1 = full resolution)
    thklim : float
        Minimum thickness to add (prevents zero thickness)

    Returns
    -------
    dict with keys:
        x, y : 1D coordinate arrays
        bed : 2D bed topography
        thickness : 2D ice thickness
        surface : 2D surface elevation
        dx : grid spacing
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required: pip install netCDF4")

    dataset = nc.Dataset(path, 'r')

    x = dataset.variables['x'][::skip]
    y = dataset.variables['y'][::skip]
    bed = dataset.variables['bed'][::skip, ::skip]
    thickness = dataset.variables['thickness'][::skip, ::skip]

    # Add minimum thickness
    thickness = thickness + thklim
    surface = bed + thickness

    dx = float(abs(x[1] - x[0]))

    dataset.close()

    return {
        'x': np.array(x),
        'y': np.array(y),
        'bed': np.array(bed),
        'thickness': np.array(thickness),
        'surface': np.array(surface),
        'dx': dx
    }


def load_velocity_mosaic(u_path, v_path):
    """
    Load velocity mosaic from GeoTIFF files.

    Parameters
    ----------
    u_path : str or Path
        Path to x-velocity GeoTIFF
    v_path : str or Path
        Path to y-velocity GeoTIFF

    Returns
    -------
    dict with keys:
        u, v : 2D velocity arrays
        x, y : 1D coordinate arrays
        bounds : (left, bottom, right, top)
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")

    with rasterio.open(u_path) as src:
        u = src.read().squeeze()
        bounds = src.bounds
        ny, nx = u.shape
        x = np.linspace(bounds.left, bounds.right, nx)
        y = np.linspace(bounds.top, bounds.bottom, ny)

    with rasterio.open(v_path) as src:
        v = src.read().squeeze()

    # Replace no-data values
    nodata = u[0, 0]
    u[u == nodata] = 0.0
    v[v == nodata] = 0.0

    return {
        'u': u.astype(np.float32),
        'v': v.astype(np.float32),
        'x': x,
        'y': y,
        'bounds': bounds
    }


def load_smb_mar(path):
    """
    Load surface mass balance from MAR climate model output.

    Parameters
    ----------
    path : str or Path
        Path to MAR netCDF file

    Returns
    -------
    dict with keys:
        smb : 2D SMB array (m/yr ice equivalent)
        x, y : 1D coordinate arrays
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 required: pip install netCDF4")

    dataset = nc.Dataset(path, 'r')

    x = np.array(dataset.variables['x'])
    y = np.array(dataset.variables['y'])
    smb = np.array(dataset.variables['SMB'][:].squeeze()) / 1000.0  # Convert to m/yr

    # Replace masked values with typical ablation
    smb[smb == smb.max()] = -2.2

    dataset.close()

    return {
        'smb': smb.astype(np.float32),
        'x': x,
        'y': y
    }


def interpolate_to_grid(data, x_data, y_data, x_target, y_target):
    """
    Interpolate 2D data to a target grid.

    Parameters
    ----------
    data : ndarray
        2D data array
    x_data, y_data : ndarray
        1D coordinate arrays for data
    x_target, y_target : ndarray
        1D coordinate arrays for target grid

    Returns
    -------
    ndarray
        Interpolated data on target grid
    """
    try:
        import cupy as cp
        from cupyx.scipy.interpolate import RegularGridInterpolator
        use_gpu = True
    except ImportError:
        from scipy.interpolate import RegularGridInterpolator
        use_gpu = False

    if use_gpu:
        x_data = cp.asarray(x_data)
        y_data = cp.asarray(y_data)
        data = cp.asarray(data)
        x_target = cp.asarray(x_target)
        y_target = cp.asarray(y_target)

    X, Y = (cp.meshgrid if use_gpu else np.meshgrid)(x_target, y_target)

    interp = RegularGridInterpolator(
        (y_data, x_data), data,
        bounds_error=False, fill_value=0.0
    )

    result = interp((Y, X))

    if use_gpu:
        return result.astype(cp.float32)
    return result.astype(np.float32)


def prepare_greenland_grid(geometry_data, n_levels=5):
    """
    Prepare grid dimensions compatible with multigrid hierarchy.

    Adjusts grid to be divisible by 2^n_levels by centering a subregion.

    Parameters
    ----------
    geometry_data : dict
        Output from load_bedmachine_greenland
    n_levels : int
        Number of multigrid levels

    Returns
    -------
    dict with adjusted arrays and metadata
    """
    x = geometry_data['x']
    y = geometry_data['y']
    factor = 2 ** n_levels

    nx_target = (len(x) // factor) * factor
    ny_target = (len(y) // factor) * factor

    # Center the subregion
    x_start = (len(x) - nx_target) // 2
    y_start = (len(y) - ny_target) // 2

    x_slice = slice(x_start, x_start + nx_target)
    y_slice = slice(y_start, y_start + ny_target)

    return {
        'x': x[x_slice],
        'y': y[y_slice],
        'bed': geometry_data['bed'][y_slice, x_slice],
        'thickness': geometry_data['thickness'][y_slice, x_slice],
        'surface': geometry_data['surface'][y_slice, x_slice],
        'dx': geometry_data['dx'],
        'ny': ny_target,
        'nx': nx_target
    }

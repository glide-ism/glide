"""
Data download and loading utilities.

Provides functions to download standard ice sheet datasets and load them
into GLIDE-compatible format. Uses gdown for automatic downloading from
Google Drive and xarray for data loading.
"""

import os
from pathlib import Path
import numpy as np

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

# =============================================================================
# File registry with Google Drive file IDs
# =============================================================================

# Google Drive file IDs (extracted from share URLs)
FILE_IDS = {
    "GLIDE_greenland_inputs.h5": "1Iu3Xro_0b8TVht7OHe2KFOrJOX4YC-nj",
    "GLIDE_antarctica_inputs.h5": "12pc5Yhmldwp6fD0-CRdXVCbrmDvg9afo",
    "bitterroot_dem.tif": "18tsQDV6D4ri7hvhYsad1Ft_c2gYb1ziE",
}


def _gdrive_url(file_id):
    """Convert Google Drive file ID to direct download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def fetch(filename, cache_dir=None, quiet=False):
    """
    Fetch a file, downloading from Google Drive if needed.

    Files are cached to a local 'data' directory by default, making them
    visible and easy to manage alongside example scripts.

    Parameters
    ----------
    filename : str
        Name of file in registry
    cache_dir : str or Path, optional
        Directory to cache files (default: ./data)
    quiet : bool
        Suppress download progress output

    Returns
    -------
    str
        Path to local file
    """
    if not HAS_GDOWN:
        raise ImportError("gdown required for automatic downloads: pip install gdown")

    # Default to ./data in current working directory
    if cache_dir is None:
        cache_dir = Path("./data")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    filepath = cache_dir / filename
    if filepath.exists():
        return str(filepath)

    # Get file ID from registry
    file_id = FILE_IDS.get(filename)
    if file_id is None:
        raise ValueError(f"Unknown file: {filename}. Available: {list(FILE_IDS.keys())}")

    # Download using gdown (handles large file confirmation automatically)
    url = _gdrive_url(file_id)
    gdown.download(url, str(filepath), quiet=quiet)
    return str(filepath)


def clear_cache(cache_dir=None):
    """Clear cached data files.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to clear (default: ./data)
    """
    import shutil
    if cache_dir is None:
        cache_dir = Path("./data")
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")


# =============================================================================
# High-level loaders for preprocessed data
# =============================================================================

def load_greenland_preprocessed(filename="GLIDE_greenland_inputs.h5", cache_dir=None, quiet=False):
    """
    Load preprocessed Greenland datasets (auto-downloads if needed).

    Parameters
    ----------
    filename : str
        Name of file in registry
    cache_dir : str or Path, optional
        Directory to cache files (default: ./data)
    quiet : bool
        Suppress download progress output

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr
    return xr.open_dataset(fetch(filename, cache_dir=cache_dir, quiet=quiet))


def load_antarctica_preprocessed(filename="GLIDE_antarctica_inputs.h5", cache_dir=None, quiet=False):
    """
    Load preprocessed Antarctica datasets (auto-downloads if needed).

    Parameters
    ----------
    filename : str
        Name of file in registry
    cache_dir : str or Path, optional
        Directory to cache files (default: ./data)
    quiet : bool
        Suppress download progress output

    Returns
    -------
    xarray.Dataset
    """
    import xarray as xr
    return xr.open_dataset(fetch(filename, cache_dir=cache_dir, quiet=quiet))


def load_bitterroot_dem(filename="bitterroot_dem.tif", cache_dir=None, quiet=False):
    """
    Load Bitterroot DEM (auto-downloads if needed).

    Parameters
    ----------
    filename : str
        Name of file in registry
    cache_dir : str or Path, optional
        Directory to cache files (default: ./data)
    quiet : bool
        Suppress download progress output

    Returns
    -------
    xarray.DataArray
    """
    import xarray as xr
    return xr.open_dataarray(fetch(filename, cache_dir=cache_dir, quiet=quiet)).squeeze()


# =============================================================================
# Legacy download function (for manual URLs)
# =============================================================================

def download_file(url, filename, cache_dir=None, force=False):
    """
    Download a file with caching (legacy function).

    Parameters
    ----------
    url : str
        URL to download from
    filename : str
        Local filename to save as
    cache_dir : Path, optional
        Cache directory (default: ./data)
    force : bool
        Re-download even if file exists

    Returns
    -------
    Path
        Path to downloaded file
    """
    import urllib.request

    if cache_dir is None:
        cache_dir = Path("./data")
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


# =============================================================================
# Loaders for raw/source data files
# =============================================================================


def load_bedmachine(path, skip=1, thklim=0.1, bbox_pad=[0,1,0,1]):
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
    bbox_pad : list
        Padding to remove from edges [x_start, x_end, y_start, y_end]

    Returns
    -------
    dict with keys:
        x, y : 1D coordinate arrays
        bed : 2D bed topography
        thickness : 2D ice thickness
        surface : 2D surface elevation
        dx : grid spacing
    """
    import xarray as xr

    ds = xr.open_dataset(path)

    # Apply bounding box padding and skip
    x = ds['x'].values[bbox_pad[0]:-bbox_pad[1]][::skip]
    y = ds['y'].values[bbox_pad[2]:-bbox_pad[3]][::skip]
    bed = ds['bed'].values[bbox_pad[2]:-bbox_pad[3], bbox_pad[0]:-bbox_pad[1]][::skip, ::skip]
    thickness = ds['thickness'].values[bbox_pad[2]:-bbox_pad[3], bbox_pad[0]:-bbox_pad[1]][::skip, ::skip]
    surface = ds['surface'].values[bbox_pad[2]:-bbox_pad[3], bbox_pad[0]:-bbox_pad[1]][::skip, ::skip]

    ds.close()

    base = np.maximum(bed, surface * (1 - (1 / (1 - 0.917))))
    thickness = surface - base + thklim

    dx = float(abs(x[1] - x[0]))

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
    """
    import xarray as xr

    u_da = xr.open_dataarray(u_path).squeeze()
    v_da = xr.open_dataarray(v_path).squeeze()

    u = u_da.values
    v = v_da.values
    x = u_da.coords['x'].values
    y = u_da.coords['y'].values

    # Replace no-data values
    nodata = u[0, 0]
    u[u == nodata] = 0.0
    v[v == nodata] = 0.0

    return {
        'u': u.astype(np.float32),
        'v': v.astype(np.float32),
        'x': x,
        'y': y,
    }

def load_antarctic_velocity(U_OBS_PATH):
    import xarray as xr
    data = xr.load_dataset(U_OBS_PATH)
    return data.variables['x'].values,data.variables['y'].values,data.variables['VX'],data.variables['VY']


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
    import xarray as xr

    ds = xr.open_dataset(path)

    x = ds['x'].values
    y = ds['y'].values
    smb = ds['SMB'].values.squeeze() / 1000.0  # Convert to m/yr

    ds.close()

    # Replace masked values with typical ablation
    smb[smb == smb.max()] = -2.2

    return {
        'smb': smb.astype(np.float32),
        'x': x,
        'y': y
    }

def load_smb_racmo(SMB_PATH,x,y):
    import xarray as xr
    import numpy as np
    from pyproj import Transformer
    from scipy.interpolate import griddata

    # Load RACMO
    racmo = xr.open_dataset(SMB_PATH)
    lat = racmo.lat.values  # (rlat, rlon)
    lon = racmo.lon.values

    # Transform to EPSG:3031
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x_racmo, y_racmo = transformer.transform(lon, lat)  # both now 2D arrays

    X, Y = np.meshgrid(x, y)

    # Flatten source coordinates and data for griddata
    x_src = x_racmo.ravel()
    y_src = y_racmo.ravel()

    # Pick a single time slice to start
    #$smb = racmo.smbgl.isel(time=0, height=0).values  # (rlat, rlon)
    smb = racmo.smbgl.values.mean(axis=0).squeeze()/917*12
    smb_flat = smb.ravel()

    # Interpolate
    smb_reprojected = griddata(
        (x_src, y_src), 
        smb_flat, 
        (X, Y), 
        method='nearest')

    return smb_reprojected

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


def prepare_grid(geometry_data, n_levels=5):
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

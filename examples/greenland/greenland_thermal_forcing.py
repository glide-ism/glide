"""
Fetch and regrid real Greenland thermal forcing onto the glide grid.

Produces two NetCDF fields on the finest glide grid (EPSG:3413, 900 m):
  data/greenland_q_geo.nc  -- geothermal heat flux  [W m^-2]
                              (Martos et al. 2018, PANGAEA 10.1594/PANGAEA.892973)
  data/greenland_tsurf.nc  -- annual-mean surface temperature [K]
                              (SeaRISE Greenland_5km_v1.1 `surftemp`, UMontana/UCAR)

Both sources are openly downloadable (no login).  Run once before
greenland_thermal_forward.py, which loads these files if present and otherwise
falls back to a uniform Q_geo and a lapse-rate surface BC.

    python greenland_thermal_forcing.py
"""
from pathlib import Path
import numpy as np
import requests
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata, RegularGridInterpolator

from glide.data import load_greenland_preprocessed

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

MARTOS_URL = ("https://store.pangaea.de/Publications/Martos-etal_2018/"
              "Geothermal_Heat_Flux_Greenland.xyz")
SEARISE_URL = ("https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/glc/cism/"
               "IceSheetData_UMontana/PresentDayGreenland/Greenland_5km_v1.1.nc")


def _download(url, dest):
    if dest.exists():
        return dest
    print(f"  downloading {dest.name} ...")
    r = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def main():
    ds = load_greenland_preprocessed()
    x, y = ds.x.values, ds.y.values                 # EPSG:3413 cell centres
    Xg, Yg = np.meshgrid(x, y)                       # (ny, nx)

    # --- Geothermal heat flux: Martos 2018 (UTM23N XYZ, mW/m^2) ---
    d = np.loadtxt(_download(MARTOS_URL, DATA / "Geothermal_Heat_Flux_Greenland.xyz"))
    xs, ys = Transformer.from_crs(32623, 3413, always_xy=True).transform(d[:, 0], d[:, 1])
    hf = d[:, 2] / 1000.0                            # mW/m^2 -> W/m^2
    q_lin = griddata((xs, ys), hf, (Xg, Yg), method="linear")
    q_near = griddata((xs, ys), hf, (Xg, Yg), method="nearest")   # fill outside the hull
    q_geo = np.where(np.isnan(q_lin), q_near, q_lin).astype(np.float32)
    xr.DataArray(q_geo, dims=("y", "x"), coords={"y": y, "x": x}, name="q_geo",
                 attrs={"units": "W m-2", "source": "Martos et al. 2018 (PANGAEA 892973)",
                        "long_name": "Geothermal heat flux"}
                 ).to_netcdf(DATA / "greenland_q_geo.nc")
    print(f"  q_geo (W/m^2): {q_geo.min():.4f}..{q_geo.max():.4f} mean {q_geo.mean():.4f}")

    # --- Surface temperature: SeaRISE surftemp (polar-stereo lat_ts=71/lon_0=-39, deg C) ---
    sr = xr.open_dataset(_download(SEARISE_URL, DATA / "Greenland_5km_v1.1.nc"),
                         engine="scipy", decode_times=False)
    x1, y1 = sr["x1"].values.astype(float), sr["y1"].values.astype(float)
    Ts_K = (np.asarray(sr["surftemp"].squeeze()) + 273.15).astype(np.float32)
    srproj = ("+proj=stere +lat_0=90 +lat_ts=71 +lon_0=-39 +x_0=0 +y_0=0 "
              "+ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    xsr, ysr = Transformer.from_crs(3413, srproj, always_xy=True).transform(Xg.ravel(), Yg.ravel())
    xsr = np.clip(xsr, x1.min(), x1.max())          # nearest-edge for the few out-of-grid cells
    ysr = np.clip(ysr, y1.min(), y1.max())
    interp = RegularGridInterpolator((y1, x1), Ts_K, method="linear", bounds_error=False, fill_value=None)
    t_surf = interp(np.column_stack([ysr, xsr])).reshape(Xg.shape).astype(np.float32)
    xr.DataArray(t_surf, dims=("y", "x"), coords={"y": y, "x": x}, name="t_surf",
                 attrs={"units": "K", "source": "SeaRISE Greenland_5km_v1.1 surftemp",
                        "long_name": "Annual mean surface temperature"}
                 ).to_netcdf(DATA / "greenland_tsurf.nc")
    print(f"  t_surf (K): {t_surf.min():.1f}..{t_surf.max():.1f} mean {t_surf.mean():.1f}")
    print("done.")


if __name__ == "__main__":
    main()

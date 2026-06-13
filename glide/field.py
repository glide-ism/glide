from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, TYPE_CHECKING
from enum import Enum
from glide.backend import xp as cp, asnumpy
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .grid import Grid

class GridEntity(str, Enum):
    CELL = "cell"
    VERTICAL_FACET = "vertical_facet"
    HORIZONTAL_FACET = "horizontal_facet"

def _to_numpy(a: Any) -> np.ndarray:
    """Convert NumPy/CuPy-like input to a NumPy array for xarray."""
    if cp is not None and isinstance(a, cp.ndarray):
        return asnumpy(a)
    return np.asarray(a)

def _maybe_scalar(a: Any) -> bool:
    arr = _to_numpy(a)
    return arr.ndim == 0

def _coord(name, values):
    if name[0] == "x":
        return (name, values, {
            "standard_name": "projection_x_coordinate",
            "units": "m",
            "axis": "X",
        })
    elif name[0] == "y":
        return (name, values, {
            "standard_name": "projection_y_coordinate",
            "units": "m",
            "axis": "Y",
        })

@dataclass
class Field:
    data: Any
    grid_entity: GridEntity
    dx: cp.float32
    grid: Grid | None = field(default=None, repr=False, compare=False)
    name: str | None = None
    units: str | None = None
    attrs: dict = field(default_factory=dict)
    _grad: Any | None = None

    def set(self, value) -> None:
        if hasattr(value, "shape"):
            self.data[...] = cp.array(value,dtype=cp.float32)
        else:
            self.data.fill(value)

    def zero(self) -> None:
        self.data.fill(0)

    @property
    def grad(self):
        if self._grad is None:
            self._grad = self._zeros_like()
        return self._grad

    def has_grad(self) -> bool:
        return self._grad is not None

    def zero_grad(self) -> None:
        if self._grad is not None:
            self._grad.fill(0)

    def _zeros_like(self):
        return cp.zeros_like(self.data)

    def __repr__(self):
        string = f'Field: {self.name}\n{self.data}\n{self.data.shape}, {self.data.dtype}, {self.units}'
        return string

    @property
    def compact_string(self):
        string = f'Field: {self.name}, {self.units}, ({self.data.shape[0]}, {self.data.shape[1]})'
        return string

    def to_cell(
        self):
        if self.grid_entity is GridEntity.CELL:
            return self
        elif self.grid_entity is GridEntity.VERTICAL_FACET:
            cell_data = 0.5*(self.data[:,1:] + self.data[:,:-1])
            return Field(data=cell_data,
                grid_entity=GridEntity.CELL,
                dx=self.dx,
                grid=self.grid,
                units=self.units,
                attrs=self.attrs)
        elif self.grid_entity is GridEntity.HORIZONTAL_FACET:
            cell_data = 0.5*(self.data[1:] + self.data[:-1])
            return Field(data=cell_data,
                grid_entity=GridEntity.CELL,
                dx=self.dx,
                grid=self.grid,
                units=self.units,
                attrs=self.attrs)
        else:
            return

    def to_dataarray(
        self,
        grid: Grid | None = None,
        *,
        name: str | None = None,
        copy: bool = False,
    ) -> xr.DataArray:
        """
        Convert this field to an xarray.DataArray.

        Priority for spatial metadata:
            1. explicit `grid` argument
            2. `self.grid`
            3. fallback to index coordinates

        Parameters
        ----------
        grid
            Optional grid override. If omitted, uses `self.grid` if available.
        name
            Optional name override for the DataArray.
        copy
            If True, copy the NumPy data before constructing the DataArray.

        Returns
        -------
        xarray.DataArray
        """
        g = grid if grid is not None else self.grid
        data = _to_numpy(self.data)
        if copy:
            data = data.copy()

        da_name = name if name is not None else self.name

        attrs: dict[str, Any] = dict(self.attrs)
        if self.units is not None:
            attrs["units"] = self.units
        if "long_name" in self.attrs:
            attrs["long_name"] = self.attrs["long_name"]
        attrs["grid_entity"] = self.grid_entity.value

        # Optional CRS metadata from the grid
        if g is not None and getattr(g, "crs", None) is not None:
            attrs["crs"] = str(g.crs)
            attrs["spatial_ref"] = g.crs.to_wkt()
            attrs["crs_wkt"] = g.crs.to_wkt()

        # Rich coordinate mode: use grid-provided coordinates
        
        if self.grid_entity is GridEntity.CELL:
            dims = ("y_cell", "x_cell")
            if g is not None:   
                coords = {
                    "x_cell": _coord("x_cell",_to_numpy(g.x_cell)),
                    "y_cell": _coord("y_cell",_to_numpy(g.y_cell)),
                }
            else:
                coords = {
                    "x_cell": _coord("x_cell",np.arange( self.dx/2, self.dx/2 + data.shape[1]*self.dx, self.dx)),
                    "y_cell": _coord("y_cell",np.arange(-self.dx/2,-self.dx/2 - data.shape[0]*self.dx,-self.dx)),
                }

        elif self.grid_entity is GridEntity.VERTICAL_FACET:
            dims = ("y_cell", "x_facet")
            if g is not None:
                coords = {
                    "x_facet": _coord("x_facet",_to_numpy(g.x_vfacet)),
                    "y_cell": _coord("y_cell",_to_numpy(g.y_cell)),
                }
            else:
                coords = {
                    "x_facet": _coord("x_facet",np.arange(0, data.shape[1]*self.dx,self.dx)),
                    "y_cell": _coord("y_cell",np.arange(-self.dx/2,-self.dx/2 - data.shape[0]*self.dx,-self.dx)),
                }

        elif self.grid_entity is GridEntity.HORIZONTAL_FACET:
            dims = ("y_facet", "x_cell")
            if g is not None:
                coords = {
                    "x_cell": _coord("x_cell",_to_numpy(g.x_cell)),
                    "y_facet": _coord("y_facet",_to_numpy(g.y_hfacet)),
                }
            else:
                coords = {
                    "x_cell": _coord("x_cell",np.arange( self.dx/2, self.dx/2 + data.shape[1]*self.dx, self.dx)),
                    "y_facet": _coord("y_facet",np.arange(0, -self.data.shape[0]*self.dx,-self.dx)),
                }

        return xr.DataArray(
            data=data,
            dims=dims,
            coords=coords,
            name=da_name,
            attrs=attrs,
            )

@dataclass
class Constant:
    value: Any
    name: str | None = None
    units: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    _grad: Any | None = None

    def set(self, value) -> None:
        self.value = cp.float32(value)

    @property
    def grad(self):
        if self._grad is None:
            self._grad = 0.0
        return self._grad

    @grad.setter
    def grad(self,value):
        self._grad = cp.float32(value)

    def has_grad(self) -> bool:
        return self._grad is not None

    def zero_grad(self) -> None:
        if self._grad is not None:
            self._grad = 0.0

    def __repr__(self):
        string = f'Constant: {self.name}, {self.value:.3f}, {self.units}'
        return string

    def to_dataarray(self):
        return xr.DataArray(
            data=self.value,
            name=self.name,
            attrs={
                "units": self.units,
                "long_name": self.attrs['long_name'] if 'long_name' in self.attrs else None,
            },
        )

@dataclass
class SubgridField(Field):
    quantiles: Any | None = None


@dataclass
class TimeField(Field):
    dt: cp.float32 | None = None
    def to_cell(
        self):
        raise AttributeError('Only supported for GridEntity.CELL')

    def to_dataarray(
        self,
        grid: Grid | None = None,
        *,
        name: str | None = None,
        copy: bool = False,
    ) -> xr.DataArray:
        raise NotImplementedError

class LocalOption:
    def __init__(self, getter, setter, name: str):
        self._getter = getter
        self._setter = setter
        self._name = name

    def get(self):
        return self._getter()

    def set(self, value):
        self._setter(value)

    def __repr__(self):
        return f"{self._name}={self.get()!r}"


class BroadcastOption:
    def __init__(self, levels, getter, attr_name: str):
        self._levels = levels
        self._getter = getter
        self._attr_name = attr_name

    def get(self):
        # Representative value from the first level
        return getattr(self._getter(self._levels[0]), self._attr_name)

    def set(self, value):
        for lev in self._levels:
            cfg = self._getter(lev)
            setattr(cfg, self._attr_name, value)

    def __repr__(self):
        return f"{self._attr_name}={self.get()!r}"

"""
Input/output utilities for GLIDE.

Provides VTI (ParaView) and HDF5 output writers for visualization and analysis.
"""
from __future__ import annotations

import numpy as np
from glide.backend import xp as cp, asnumpy
import xarray as xr
import zarr
import xml.etree.ElementTree as ET
import shutil

from xml.dom import minidom
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping

def _pretty_xml(element):
    """Format XML element with indentation."""
    return minidom.parseString(ET.tostring(element)).toprettyxml(indent="  ")


def write_vti(filename, data, dx, dy=None, origin=(0.0, 0.0), time_value=None, flip_y=True):
    """
    Write fields to VTI (VTK ImageData) binary format.

    Parameters
    ----------
    filename : str or Path
        Output filename
    data : dict
        Dictionary mapping field names to data. Values can be:
        - CuPy/NumPy array: scalar field
        - List of arrays: vector field components
    dx : float
        Grid spacing in x
    dy : float, optional
        Grid spacing in y (defaults to dx)
    origin : tuple
        Grid origin (x, y)
    time_value : float, optional
        Time value for this snapshot
    flip_y : bool
        Flip arrays along y-axis (convert from image to VTK convention)
    """
    if dy is None:
        dy = dx

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Separate scalars and vectors, transfer to CPU
    scalars = {}
    vectors = {}

    for name, value in data.items():
        if isinstance(value, list):
            components = [asnumpy(c).astype(np.float32) for c in value]
            if flip_y:
                components = [np.flip(c, axis=0) for c in components]
            vectors[name] = components
        else:
            arr = asnumpy(value).astype(np.float32)
            if flip_y:
                arr = np.flip(arr, axis=0)
            scalars[name] = arr

    # Get grid dimensions
    if scalars:
        first_field = next(iter(scalars.values()))
    else:
        first_field = next(iter(vectors.values()))[0]
    ny, nx = first_field.shape

    # Build XML
    root = ET.Element("VTKFile", type="ImageData", version="1.0", byte_order="LittleEndian")
    img = ET.SubElement(root, "ImageData",
                        WholeExtent=f"0 {nx-1} 0 {ny-1} 0 0",
                        Origin=f"{origin[0]} {origin[1]} 0",
                        Spacing=f"{dx} {dy} 1.0")
    piece = ET.SubElement(img, "Piece", Extent=f"0 {nx-1} 0 {ny-1} 0 0")

    if time_value is not None:
        fd = ET.SubElement(piece, "FieldData")
        da = ET.SubElement(fd, "DataArray",
                           type="Float32", Name="TimeValue",
                           NumberOfComponents="1", format="ascii")
        da.text = f"\n{float(time_value)}\n"

    pd_attrs = {}
    if scalars:
        pd_attrs["Scalars"] = next(iter(scalars.keys()))
    for name, components in vectors.items():
        if len(components) == 3:
            pd_attrs["Vectors"] = name
            break
    pd = ET.SubElement(piece, "PointData", **pd_attrs)

    binary_arrays = []
    offset = 0

    for name, arr in scalars.items():
        arr_bytes = arr.ravel(order='C').tobytes()
        ET.SubElement(pd, "DataArray",
                      type="Float32", Name=name,
                      NumberOfComponents="1",
                      format="appended",
                      offset=str(offset))
        binary_arrays.append(arr_bytes)
        offset += len(arr_bytes) + 4

    for name, components in vectors.items():
        ncomp = len(components)
        stacked = np.stack(components, axis=-1).astype(np.float32)
        vec_bytes = stacked.ravel(order='C').tobytes()
        ET.SubElement(pd, "DataArray",
                      type="Float32", Name=name,
                      NumberOfComponents=str(ncomp),
                      format="appended",
                      offset=str(offset))
        binary_arrays.append(vec_bytes)
        offset += len(vec_bytes) + 4

    xml_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)

    with open(filename, 'wb') as f:
        xml_str = xml_bytes.decode('utf-8')
        if xml_str.endswith('</VTKFile>'):
            xml_str = xml_str[:-len('</VTKFile>')]
        elif xml_str.endswith('</VTKFile>\n'):
            xml_str = xml_str[:-len('</VTKFile>\n')]

        f.write(xml_str.encode('utf-8'))
        f.write(b'  <AppendedData encoding="raw">\n   _')

        for array_bytes in binary_arrays:
            size = np.uint32(len(array_bytes))
            f.write(size.tobytes())
            f.write(array_bytes)

        f.write(b'\n  </AppendedData>\n</VTKFile>\n')


@dataclass
class VTIWriter:
    """
    Write static and time-dependent VTI outputs with a PVD manifest.

    The public interface intentionally mirrors :class:`ZarrWriter` as closely as
    possible:

    - configure ``static_fields`` and ``dynamic_fields`` at construction time
    - call :meth:`initialize` once
    - call :meth:`append` for each dynamic snapshot

    Field mappings may contain either scalar fields/constants or vectors. Vectors
    are specified as lists/tuples of fields, e.g. ``{"U": [grid.state.u,
    grid.state.v]}``. Vector components are collocated to cell centers via each
    field's ``to_cell()`` method before writing.
    """

    out_dir: str | Path
    static_fields: Mapping[str, Any] = field(default_factory=dict)
    dynamic_fields: Mapping[str, Any] = field(default_factory=dict)
    base: str = "output"
    dx: float = 1.0
    dy: float | None = None
    origin: tuple[float, float] = (0.0, 0.0)
    flip_y: bool = True

    _step_idx: int = field(default=0, init=False, repr=False)
    _static_written: bool = field(default=False, init=False, repr=False)
    records: list[tuple[float, str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.dy is None:
            self.dy = self.dx

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------

    def initialize(
        self,
        grid: Any,
        *,
        overwrite: bool = False,
        write_dynamic_initial: bool = False,
        time: Any | None = None,
    ) -> None:
        """
        Initialize the output directory and optionally write static fields and an
        initial dynamic snapshot.
        """
        if overwrite:
            for path in self.out_dir.glob(f"{self.base}_*.vti"):
                path.unlink()
            pvd_path = self.out_dir / f"{self.base}.pvd"
            if pvd_path.exists():
                pvd_path.unlink()
            self.records.clear()
            self._step_idx = 0
            self._static_written = False

        if self.static_fields:
            self.write_static(grid, overwrite=overwrite)

        if write_dynamic_initial:
            if time is None:
                raise ValueError(
                    "Argument `time` must be provided when write_dynamic_initial=True."
                )
            self.append(grid, time=time)

    def append(self, grid: Any, *, time: Any) -> Path:
        """Append one dynamic snapshot at the given time."""
        if not self.dynamic_fields:
            raise ValueError("No dynamic fields were configured for this writer.")
        return self.append_dynamic(grid, time=time)

    # ------------------------------------------------------------------
    # Lower-level API
    # ------------------------------------------------------------------

    def write_static(self, grid: Any, *, overwrite: bool = False) -> Path | None:
        """Write configured static fields once to ``{base}_static.vti``."""
        if not self.static_fields:
            return None
        if self._static_written and not overwrite:
            return self.out_dir / f"{self.base}_static.vti"

        data = self._build_vti_payload(grid, self.static_fields)
        fpath = self.out_dir / f"{self.base}_static.vti"
        write_vti(
            fpath,
            data,
            self.dx,
            self.dy,
            self.origin,
            time_value=None,
            flip_y=self.flip_y,
        )
        self._static_written = True
        return fpath

    def append_dynamic(self, grid: Any, *, time: Any) -> Path:
        """Append one time slice of configured dynamic variables."""
        data = self._build_vti_payload(grid, self.dynamic_fields)
        fpath = self.write_step(self._step_idx, time, data)
        self._step_idx += 1
        return fpath

    # ------------------------------------------------------------------
    # Compatibility API
    # ------------------------------------------------------------------

    def write_step(self, step_idx: int, time_value: Any, data: Mapping[str, Any]) -> Path:
        """Write a timestep to a numbered VTI file."""
        fname = f"{self.base}_{step_idx:04d}.vti"
        fpath = self.out_dir / fname
        write_vti(
            fpath,
            data,
            self.dx,
            self.dy,
            self.origin,
            time_value=time_value,
            flip_y=self.flip_y,
        )
        self.records.append((float(time_value), fname))
        return fpath

    def write_pvd(self, pvd_name: str | None = None) -> Path:
        """Write the PVD manifest file."""
        if pvd_name is None:
            pvd_name = f"{self.base}.pvd"

        root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        coll = ET.SubElement(root, "Collection")
        for t, fname in self.records:
            ET.SubElement(
                coll,
                "DataSet",
                timestep=str(t),
                group="",
                part="0",
                file=str(fname),
            )

        with open(self.out_dir / pvd_name, "w") as f:
            f.write(_pretty_xml(root))
        return self.out_dir / pvd_name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vti_payload(self, grid: Any, fields: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, value in fields.items():
            payload[name] = self._coerce_vti_value(value)
        return payload

    def _coerce_vti_value(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [self._coerce_vector_component(component) for component in value]
        return self._coerce_scalar_value(value)

    def _coerce_scalar_value(self, value: Any) -> Any:
        if hasattr(value, "to_cell"):
            return value.to_cell().data
        if hasattr(value, "data"):
            return value.data
        return value

    def _coerce_vector_component(self, value: Any) -> Any:
        if hasattr(value, "to_cell"):
            return value.to_cell().data
        if hasattr(value, "data"):
            return value.data
        return value


@dataclass
class ZarrWriter:
    """
    Write static and time-dependent simulation outputs to a Zarr store.

    Typical usage
    -------------
    writer = ZarrWriter(
        "run.zarr",
        static_fields={
            "bed": grid.geometry.bed,
            "beta": grid.physics.sliding.beta,
            "thklim": grid.geometry.thklim,
        },
        dynamic_fields={
            "H": grid.state.H,
            "u": grid.state.u,
            "v": grid.state.v,
            "mask": grid.state.mask,
        },
    )

    writer.initialize(grid)

    for t in times:
        ...
        writer.append(grid, time=t)

    Notes
    -----
    - Static variables are written once, without a `time` dimension.
    - Dynamic variables are written with a leading `time` dimension and appended.
    - `grid.to_dataset(fields=...)` is assumed to return an xarray.Dataset.
    """

    store: str | Path
    static_fields: Mapping[str, Any] = field(default_factory=dict)
    dynamic_fields: Mapping[str, Any] = field(default_factory=dict)

    time_dim: str = "time"
    mode_static: str = "a"
    mode_dynamic_first: str = "a"
    mode_dynamic_append: str = "a"
    consolidated: bool = False

    static_encoding: dict[str, dict[str, Any]] = field(default_factory=dict)
    dynamic_encoding: dict[str, dict[str, Any]] = field(default_factory=dict)

    _static_written: bool = field(default=False, init=False, repr=False)
    _dynamic_initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.store = str(self.store)

    # -------------------------------------------------------------------------
    # Public high-level API
    # -------------------------------------------------------------------------

    def initialize(
        self,
        grid: Any,
        *,
        attrs: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        write_dynamic_initial: bool = False,
        time: Any | None = None,
    ) -> None:
        """
        Initialize the Zarr store.

        Parameters
        ----------
        overwrite
            If True, delete any existing store and start fresh.
        """

        # --- NEW: hard reset store if requested ---
        if overwrite:
            path = Path(self.store)
            if path.exists():
                shutil.rmtree(path)

            # reset internal flags
            self._static_written = False
            self._dynamic_initialized = False

        # --- determine correct mode for first write ---
        first_write_mode = "w" if overwrite else self.mode_static

        # --- write static fields ---
        if self.static_fields:
            ds_static = self._build_dataset(grid, self.static_fields, attrs=attrs)

            if self.time_dim in ds_static.dims:
                raise ValueError(
                    f"Static dataset unexpectedly contains time dimension '{self.time_dim}'."
                )

            self._validate_dataset_names(ds_static)

            ds_static.to_zarr(
                self.store,
                mode=first_write_mode,
                consolidated=self.consolidated,
                encoding=self.static_encoding or None,
            )

            self._static_written = True

            # after first write, revert to append mode
            self.mode_static = "a"

        # --- optionally write first dynamic snapshot ---
        if write_dynamic_initial:
            if time is None:
                raise ValueError(
                    "Argument `time` must be provided when write_dynamic_initial=True."
                )

            # if no static write happened, we need to create the store here
            if not self.static_fields:
                self.mode_dynamic_first = "w" if overwrite else self.mode_dynamic_first

            self.append(grid, time=time, attrs=attrs)

    def append(
        self,
        grid: Any,
        *,
        time: Any,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Append one dynamic snapshot at the given time.
        """
        if not self.dynamic_fields:
            raise ValueError("No dynamic fields were configured for this writer.")

        self.append_dynamic(grid, time=time, attrs=attrs)

    # -------------------------------------------------------------------------
    # Lower-level API
    # -------------------------------------------------------------------------

    def write_static(
        self,
        grid: Any,
        *,
        attrs: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Write configured static variables once.
        """
        if not self.static_fields:
            return

        if self._static_written and not overwrite:
            return

        ds = self._build_dataset(grid, self.static_fields, attrs=attrs)

        if self.time_dim in ds.dims:
            raise ValueError(
                f"Static dataset unexpectedly contains time dimension '{self.time_dim}'."
            )

        self._validate_dataset_names(ds)

        ds.to_zarr(
            self.store,
            mode=self.mode_static,
            consolidated=self.consolidated,
            encoding=self.static_encoding or None,
        )

        self._static_written = True

    def append_dynamic(
        self,
        grid: Any,
        *,
        time: Any,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Append one time slice of configured dynamic variables.
        """
        ds = self._build_dataset(grid, self.dynamic_fields, attrs=attrs)

        if self.time_dim in ds.dims:
            raise ValueError(
                f"Dynamic dataset should not already contain time dimension '{self.time_dim}'. "
                f"The writer adds it automatically."
            )

        ds = ds.expand_dims({self.time_dim: [time]})

        self._validate_has_time(ds)
        self._validate_dataset_names(ds)

        if not self._dynamic_initialized:
            ds.to_zarr(
                self.store,
                mode=self.mode_dynamic_first,
                consolidated=self.consolidated,
                encoding=self.dynamic_encoding or None,
            )
            self._dynamic_initialized = True
        else:
            ds.to_zarr(
                self.store,
                mode=self.mode_dynamic_append,
                append_dim=self.time_dim,
                consolidated=self.consolidated,
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_dataset(
        self,
        grid: Any,
        fields: Mapping[str, Any],
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> xr.Dataset:
        ds = grid.to_dataset(fields=fields)

        if attrs:
            ds = ds.copy()
            ds.attrs.update(dict(attrs))

        return ds

    def _validate_has_time(self, ds: xr.Dataset) -> None:
        if self.time_dim not in ds.dims:
            raise ValueError(
                f"Dataset does not contain required time dimension '{self.time_dim}'."
            )

    def _validate_dataset_names(self, ds: xr.Dataset) -> None:
        if self.time_dim in ds.data_vars:
            raise ValueError(
                f"Dataset contains a data variable named '{self.time_dim}', which is reserved "
                f"for the time coordinate."
            )

    def consolidate_metadata(self) -> None:
        zarr.consolidate_metadata(self.store)

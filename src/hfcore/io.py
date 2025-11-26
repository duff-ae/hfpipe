# src/hfcore/io.py
from __future__ import annotations

import os
import glob
from typing import Iterable, Mapping, Any, Dict

import numpy as np
import tables

from .hd5schema import DefaultLumitable, open_hd5
from .decorators import log_step, timeit

# Column names defined by the default lumi table schema.
LUMITABLE_COLUMNS = tuple(DefaultLumitable.columns.keys())

# ----------------------------------------------------------------------
#  Writing
# ----------------------------------------------------------------------
@log_step("save_to_hd5")
@timeit("save_to_hd5")
def save_to_hd5(
    rows: Iterable[Mapping[str, Any]],
    node: str,
    path: str,
    name: str,
) -> None:
    """
    Save a sequence of row dictionaries into an HDF5 table.

    Parameters
    ----------
    rows : iterable of dict
        Each element is a mapping {column_name: value}. Only columns present
        in `DefaultLumitable` are written; extra keys in the dict are ignored.
    node : str
        Name of the table node under the HDF5 root (e.g. "hfetlumi").
    path : str
        Base directory where the file will be created.
    name : str
        File name (may include relative subdirectories). The final path is
        constructed as os.path.join(path, name).

    Notes
    -----
    - All intermediate directories are created automatically.
    - The file is opened in "w" mode, i.e. any existing file with the same
      name will be overwritten.
    """
    # Full file path
    full_path = os.path.join(path, name)

    # Ensure the parent directory exists
    parent_dir = os.path.dirname(full_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Open/create HDF5 file
    h5out = open_hd5(full_path, mode="w")

    try:
        # If a node with the same name already exists, drop it first
        if hasattr(h5out.root, node):
            h5out.remove_node("/", node)

        filters = tables.Filters(complevel=9, complib="blosc")
        chunkshape = (100,)

        outtable: tables.Table = h5out.create_table(
            "/",
            node,
            DefaultLumitable,
            filters=filters,
            chunkshape=chunkshape,
        )

        rownew = outtable.row
        for r in rows:
            # r is a dict: {column_name: value}
            for col in DefaultLumitable.columns.keys():
                if col in r:
                    rownew[col] = r[col]
            rownew.append()

        outtable.flush()
    finally:
        h5out.close()


# ----------------------------------------------------------------------
#  Reading into numpy arrays for algorithms
# ----------------------------------------------------------------------
@log_step("load_hd5_to_arrays")
@timeit("load_hd5_to_arrays")
def load_hd5_to_arrays(directory: str, pattern: str, node: str = "hfetlumi") -> dict:
    """
    Load one or more HDF5 tables `node` matching a given pattern and
    concatenate them along the row axis.

    Parameters
    ----------
    directory : str
        Base directory where the HDF5 files live, e.g.
        "/.../hfet/25/10709".
    pattern : str
        File glob pattern relative to `directory`, e.g. "10709_*.hd5".
    node : str, optional
        Name of the HDF5 table under the root (default: "hfetlumi").

    Returns
    -------
    data : dict[str, np.ndarray]
        A dictionary mapping column names to numpy arrays. All files are
        concatenated along axis 0.

    Raises
    ------
    FileNotFoundError
        If no files matching the pattern are found.
    RuntimeError
        If a file does not contain the requested node, or if column shapes
        are inconsistent across files.
    """
    full_pattern = os.path.join(directory, pattern)
    paths = sorted(glob.glob(full_pattern))

    if not paths:
        raise FileNotFoundError(f"No files matching pattern '{full_pattern}'")

    all_data: dict[str, np.ndarray] | None = None

    for path in paths:
        h5 = open_hd5(path, mode="r")
        try:
            if not hasattr(h5.root, node):
                raise RuntimeError(f"Node '/{node}' not found in {path}")

            table: tables.Table = getattr(h5.root, node)

            # Read all columns of this table into numpy arrays
            local_data: dict[str, np.ndarray] = {}
            for colname in table.coldescrs.keys():
                col = table.col(colname)            # already a numpy array-like
                local_data[colname] = np.array(col)  # make an explicit copy

            if "bxraw" not in local_data and "data" in local_data:
                local_data["bxraw"] = local_data.pop("data")

        finally:
            h5.close()

        if all_data is None:
            # First file: just initialize
            all_data = local_data
        else:
            # Subsequent files: concatenate along axis 0
            for key, arr in local_data.items():
                if key not in all_data:
                    # New column that did not exist before: just add it
                    all_data[key] = arr
                    continue

                # Check compatibility of shapes (except for axis 0)
                if all_data[key].ndim != arr.ndim:
                    raise RuntimeError(
                        f"Column '{key}' has different ndim across files: "
                        f"{all_data[key].ndim} vs {arr.ndim}"
                    )
                if all_data[key].shape[1:] != arr.shape[1:]:
                    raise RuntimeError(
                        f"Column '{key}' has incompatible shapes across files: "
                        f"{all_data[key].shape} vs {arr.shape}"
                    )

                all_data[key] = np.concatenate([all_data[key], arr], axis=0)

    if all_data is None:
        # This should not happen given the earlier checks, but keep it explicit
        raise RuntimeError(f"No data loaded from pattern '{full_pattern}'")

    return all_data


def arrays_to_rows(data: Dict[str, np.ndarray]) -> Iterable[Dict[str, Any]]:
    """
    Convert a dictionary of numpy arrays back into an iterable of row dicts
    suitable for `save_to_hd5`.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Dictionary mapping column names to arrays. All arrays are expected
        to share the same length T along the first axis.

    Yields
    ------
    row : dict[str, Any]
        Dictionaries with keys corresponding to `LUMITABLE_COLUMNS` and
        values taken from `data[col][i]` for each row index i.
    """
    if not data:
        return

    # Determine the number of rows T from any column
    some_key = next(iter(data.keys()))
    T = data[some_key].shape[0]

    # Optional consistency check: all columns must have the same length
    for key, arr in data.items():
        if arr.shape[0] != T:
            raise RuntimeError(
                f"arrays_to_rows: column '{key}' has length {arr.shape[0]} "
                f"but expected {T}"
            )

    for i in range(T):
        row: Dict[str, Any] = {}
        for col in LUMITABLE_COLUMNS:
            arr = data[col]
            if arr.ndim == 1:
                row[col] = arr[i]
            else:
                row[col] = arr[i, ...]
        yield row
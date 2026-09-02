# src/hfcore/io.py
from __future__ import annotations

import os
import glob
from typing import Iterable, Mapping, Any, Dict, Iterator, Optional

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

    See Also
    --------
    Hd5ChunkWriter : incremental analogue for streaming/chunked pipelines,
        which avoids ever materializing the full `rows` sequence in memory.
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

    Notes
    -----
    Loads the *entire* matched dataset into memory at once. For large
    fills that don't fit comfortably in memory (e.g. on condor), use
    `iter_hd5_row_chunks` instead, which yields the same column layout
    incrementally.
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


# ----------------------------------------------------------------------
#  Chunked reading (for condor / memory-bounded processing)
# ----------------------------------------------------------------------
@log_step("iter_hd5_row_chunks")
def iter_hd5_row_chunks(
    directory: str,
    pattern: str,
    node: str = "hfetlumi",
    chunk_size: int = 5000,
    fill_filter: Optional[int] = None,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Stream row chunks from one or more HDF5 files matching `pattern`,
    without ever loading a whole file (or the whole fill) into memory.

    This is the chunked analogue of `load_hd5_to_arrays`: same column
    layout, same 'data' -> 'bxraw' rename, same glob + sort ordering
    over files. The only behavioural difference is that data is handed
    out incrementally instead of concatenated eagerly.

    Chunks never straddle a file boundary. This keeps bookkeeping simple
    and matches the per-file concatenation order of the original loader.

    Parameters
    ----------
    directory, pattern, node : same meaning as in `load_hd5_to_arrays`.
    chunk_size : int
        Number of rows per yielded chunk (last chunk of a file may be
        smaller).
    fill_filter : int, optional
        If given, rows with fillnum != fill_filter are dropped from each
        chunk before it is yielded. A chunk that becomes empty after
        filtering is skipped entirely (never yielded), so downstream
        code never has to special-case zero-length chunks.

    Yields
    ------
    dict[str, np.ndarray]
        One chunk of rows: column name -> 1D/2D numpy array of length
        <= chunk_size along axis 0.

    Raises
    ------
    FileNotFoundError
        If no files match the pattern.
    RuntimeError
        If a file does not contain the requested node.
    """
    full_pattern = os.path.join(directory, pattern)
    paths = sorted(glob.glob(full_pattern))

    if not paths:
        raise FileNotFoundError(f"No files matching pattern '{full_pattern}'")

    for path in paths:
        h5 = open_hd5(path, mode="r")
        try:
            if not hasattr(h5.root, node):
                raise RuntimeError(f"Node '/{node}' not found in {path}")

            table: tables.Table = getattr(h5.root, node)
            colnames = list(table.coldescrs.keys())
            nrows = table.nrows

            for start in range(0, nrows, chunk_size):
                stop = min(start + chunk_size, nrows)

                # table.read(start, stop) reads only this row range off
                # disk -- this is the actual memory-saving step.
                block = table.read(start=start, stop=stop)

                chunk: Dict[str, np.ndarray] = {
                    col: np.array(block[col]) for col in colnames
                }
                if "bxraw" not in chunk and "data" in chunk:
                    chunk["bxraw"] = chunk.pop("data")

                if fill_filter is not None and "fillnum" in chunk:
                    sel = (np.asarray(chunk["fillnum"]) == fill_filter)
                    if not np.any(sel):
                        continue
                    if not np.all(sel):
                        chunk = {k: v[sel] for k, v in chunk.items()}

                yield chunk
        finally:
            h5.close()


def count_hd5_rows(directory: str, pattern: str, node: str = "hfetlumi") -> int:
    """
    Total row count across all files matching `pattern`, without reading
    any actual row data. Useful for progress bars / sanity logging before
    a chunked pass.
    """
    full_pattern = os.path.join(directory, pattern)
    paths = sorted(glob.glob(full_pattern))
    total = 0
    for path in paths:
        h5 = open_hd5(path, mode="r")
        try:
            if hasattr(h5.root, node):
                total += int(getattr(h5.root, node).nrows)
        finally:
            h5.close()
    return total


class Hd5ChunkWriter:
    """
    Keeps a single output HDF5 table open across many `write_chunk()`
    calls, appending rows without ever materializing the full dataset
    in memory. Produces files with the exact same table layout as
    `save_to_hd5` (same schema, filters, chunkshape), so anything
    written this way is a drop-in replacement for the old "build a big
    dict, call save_to_hd5 once" pattern.

    Usage
    -----
        with Hd5ChunkWriter(path, node="hfetlumi") as writer:
            for chunk in iter_hd5_row_chunks(...):
                ... process chunk ...
                writer.write_chunk(chunk)
    """

    def __init__(self, path: str, node: str, complevel: int = 9):
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        self.path = path
        self.node = node
        self.h5 = open_hd5(path, mode="w")

        if hasattr(self.h5.root, node):
            self.h5.remove_node("/", node)

        filters = tables.Filters(complevel=complevel, complib="blosc")
        self.table: tables.Table = self.h5.create_table(
            "/", node, DefaultLumitable, filters=filters, chunkshape=(100,)
        )
        self._closed = False
        self._n_rows_written = 0

    @property
    def n_rows_written(self) -> int:
        return self._n_rows_written

    def write_chunk(self, data: Dict[str, np.ndarray]) -> None:
        """
        Append one chunk (dict[col] -> np.ndarray, all arrays sharing the
        same length along axis 0) to the output table.

        Builds a structured array matching the table's dtype and calls
        `table.append()` once per chunk -- much faster than the
        per-row python loop in `save_to_hd5`, and the reason chunked
        writing doesn't become the new bottleneck.
        """
        if self._closed:
            raise RuntimeError("Hd5ChunkWriter: write_chunk() called after close()")

        some_key = next(iter(data.keys()), None)
        if some_key is None:
            return

        T = data[some_key].shape[0]
        if T == 0:
            return

        for key, arr in data.items():
            if arr.shape[0] != T:
                raise RuntimeError(
                    f"Hd5ChunkWriter.write_chunk: column '{key}' has length "
                    f"{arr.shape[0]}, expected {T} (from column '{some_key}')"
                )

        buf = np.zeros(T, dtype=self.table.dtype)
        for col in self.table.dtype.names:
            if col in data:
                buf[col] = data[col]
        self.table.append(buf)
        self._n_rows_written += T

    def flush(self) -> None:
        if not self._closed:
            self.table.flush()

    def close(self) -> None:
        if not self._closed:
            self.table.flush()
            self.h5.close()
            self._closed = True

    def __enter__(self) -> "Hd5ChunkWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
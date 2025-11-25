# src/hfcore/hd5schema.py

"""
HDF5 schema and utilities for luminosity tables.

This module provides the canonical lumi table layout (`DefaultLumitable`)
and convenience wrappers for opening files and creating/fetching the table.

All modules interacting with HDF5 files should import BX_LEN and
DefaultLumitable from here to ensure consistent schema.
"""

import tables

# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------

# The single source of truth for the number of bunch crossings (BX)
BX_LEN = 3564


# ----------------------------------------------------------------------
#  Lumi table schema
# ----------------------------------------------------------------------

class DefaultLumitable(tables.IsDescription):
    """
    HDF5 table structure describing the per-LS luminosity payload.

    Column definition corresponds exactly to the CERN BRIL lumi format
    used for HFET-based datasets. All fields are flat scalars except
    `bxraw` and `bx`, which are fixed-length arrays of length BX_LEN.
    """

    fillnum       = tables.UInt32Col(shape=(),       dflt=0,    pos=0)
    runnum        = tables.UInt32Col(shape=(),       dflt=0,    pos=1)
    lsnum         = tables.UInt32Col(shape=(),       dflt=0,    pos=2)
    nbnum         = tables.UInt32Col(shape=(),       dflt=0,    pos=3)
    timestampsec  = tables.UInt32Col(shape=(),       dflt=0,    pos=4)
    timestampmsec = tables.UInt32Col(shape=(),       dflt=0,    pos=5)
    totsize       = tables.UInt32Col(shape=(),       dflt=0,    pos=6)

    publishnnb    = tables.UInt8Col(shape=(),        dflt=0,    pos=7)
    datasourceid  = tables.UInt8Col(shape=(),        dflt=0,    pos=8)
    algoid        = tables.UInt8Col(shape=(),        dflt=0,    pos=9)
    channelid     = tables.UInt8Col(shape=(),        dflt=0,    pos=10)
    payloadtype   = tables.UInt8Col(shape=(),        dflt=0,    pos=11)

    calibtag      = tables.StringCol(itemsize=32,    shape=(),  dflt='', pos=12)

    avgraw        = tables.Float32Col(shape=(),      dflt=0.0,  pos=13)
    avg           = tables.Float32Col(shape=(),      dflt=0.0,  pos=14)

    # Full per-BX arrays
    bxraw         = tables.Float32Col(shape=(BX_LEN,), dflt=0.0, pos=15)
    bx            = tables.Float32Col(shape=(BX_LEN,), dflt=0.0, pos=16)

    # Masks (bit-packed)
    maskhigh      = tables.UInt32Col(shape=(),       dflt=0,    pos=17)
    masklow       = tables.UInt32Col(shape=(),       dflt=0,    pos=18)


# ----------------------------------------------------------------------
#  Utilities
# ----------------------------------------------------------------------

def open_hd5(path: str, mode: str = "r") -> tables.File:
    """
    Open an HDF5 file using PyTables.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    mode : str, optional
        File mode ("r", "w", "a", etc.), default is read-only.

    Returns
    -------
    tables.File
        PyTables file handle.
    """
    return tables.open_file(path, mode=mode)


def get_or_create_lumi_table(h5: tables.File, node_name: str = "lumi") -> tables.Table:
    """
    Retrieve an existing lumi table (DefaultLumitable) from the given HDF5 file
    or create a new one if it doesn't exist.

    Parameters
    ----------
    h5 : tables.File
        Open HDF5 file handle.
    node_name : str, optional
        Name of the table under the root group.

    Returns
    -------
    tables.Table
        The existing or newly created lumi table.

    Raises
    ------
    TypeError
        If a node with the given name exists but is not a Table.
    """
    where = "/"

    if hasattr(h5.root, node_name):
        node = getattr(h5.root, node_name)
        if isinstance(node, tables.Table):
            return node
        else:
            raise TypeError(
                f"Node '/{node_name}' exists but is not a Table (type={type(node)})"
            )

    # Create new table
    filters = tables.Filters(complevel=9, complib="blosc")
    chunkshape = (100,)

    table = h5.create_table(
        where,
        node_name,
        DefaultLumitable,
        filters=filters,
        chunkshape=chunkshape,
    )
    return table
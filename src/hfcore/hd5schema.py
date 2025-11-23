# src/hfcore/hd5schema.py
import tables

BX_LEN = 3564  # один источник правды по количеству BX

class DefaultLumitable(tables.IsDescription):
    fillnum      = tables.UInt32Col(shape=(),      dflt=0,   pos=0)
    runnum       = tables.UInt32Col(shape=(),      dflt=0,   pos=1)
    lsnum        = tables.UInt32Col(shape=(),      dflt=0,   pos=2)
    nbnum        = tables.UInt32Col(shape=(),      dflt=0,   pos=3)
    timestampsec = tables.UInt32Col(shape=(),      dflt=0,   pos=4)
    timestampmsec= tables.UInt32Col(shape=(),      dflt=0,   pos=5)
    totsize      = tables.UInt32Col(shape=(),      dflt=0,   pos=6)
    publishnnb   = tables.UInt8Col(shape=(),       dflt=0,   pos=7)
    datasourceid = tables.UInt8Col(shape=(),       dflt=0,   pos=8)
    algoid       = tables.UInt8Col(shape=(),       dflt=0,   pos=9)
    channelid    = tables.UInt8Col(shape=(),       dflt=0,   pos=10)
    payloadtype  = tables.UInt8Col(shape=(),       dflt=0,   pos=11)
    calibtag     = tables.StringCol(itemsize=32,   shape=(), dflt='', pos=12)
    avgraw       = tables.Float32Col(shape=(),     dflt=0.0, pos=13)
    avg          = tables.Float32Col(shape=(),     dflt=0.0, pos=14)
    bxraw        = tables.Float32Col(shape=(BX_LEN,), dflt=0.0, pos=15)
    bx           = tables.Float32Col(shape=(BX_LEN,), dflt=0.0, pos=16)
    maskhigh     = tables.UInt32Col(shape=(),      dflt=0,   pos=17)
    masklow      = tables.UInt32Col(shape=(),      dflt=0,   pos=18)


def open_hd5(path: str, mode: str = "r") -> tables.File:
    """
    Открыть HD5-файл. Обёртка для единообразия.
    """
    return tables.open_file(path, mode=mode)


def get_or_create_lumi_table(h5: tables.File, node_name: str = "lumi") -> tables.Table:
    """
    Получить существующую таблицу DefaultLumitable по имени `node_name`,
    либо создать новую в корне.
    """
    where = "/"
    if hasattr(h5.root, node_name):
        node = getattr(h5.root, node_name)
        if isinstance(node, tables.Table):
            return node
        else:
            raise TypeError(f"Node '/{node_name}' is not a Table")
    else:
        filters = tables.Filters(complevel=9, complib='blosc')
        chunkshape = (100,)
        table = h5.create_table(
            where,
            node_name,
            DefaultLumitable,
            filters=filters,
            chunkshape=chunkshape,
        )
        return table
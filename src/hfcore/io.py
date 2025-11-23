# src/hfcore/io.py
from __future__ import annotations

import os
from typing import Iterable, Mapping, Any, Tuple, Dict

import numpy as np
import tables, glob

from .hd5schema import DefaultLumitable, open_hd5, get_or_create_lumi_table, BX_LEN
from .decorators import log_step, timeit

LUMITABLE_COLUMNS = tuple(DefaultLumitable.columns.keys())


# ----------------------------------------------------------------------
#  Запись
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
    Записать данные в HD5-таблицу формата DefaultLumitable.

    rows: iterable объектов, поддерживающих доступ по row[col_name]
          (dict, pandas.Series, numpy.void со dtype.names и т.п.)
    node: имя таблицы в корне (например 'lumi')
    path: директория
    name: имя файла (например 'fill_7920.h5')
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, name)

    # перезаписываем файл (как в старой версии)
    h5out = open_hd5(full_path, mode='w')
    try:
        outtable = get_or_create_lumi_table(h5out, node_name=node)
        rownew = outtable.row

        for row in rows:
            for col in LUMITABLE_COLUMNS:
                # Требуем, чтобы входной row содержал все эти ключи.
                # В случае numpy.void можно сделать row[col].
                value = row[col]
                rownew[col] = value
            rownew.append()

        outtable.flush()
    finally:
        h5out.close()


# ----------------------------------------------------------------------
#  Чтение в массивы для алгоритмов
# ----------------------------------------------------------------------
@log_step("load_hd5_to_arrays")
@timeit("load_hd5_to_arrays")
def load_hd5_to_arrays(directory: str, pattern: str, node: str = "hfetlumi") -> dict:
    """
    Загружает одну или несколько таблиц `node` из HD5-файлов под заданным паттерном
    и конкатенирует их по строкам.

    Пример:
      directory = "/.../hfet/25/10709"
      pattern   = "10709_*.hd5"

    Тогда будут прочитаны все файлы /.../hfet/25/10709/10709_*.hd5
    и таблицы hfetlumi из них будут склеены.
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
            # Считываем ВСЕ колонки этой таблицы в numpy
            local_data: dict[str, np.ndarray] = {}
            for colname in table.coldescrs.keys():
                col = table.col(colname)      # уже готовый np.ndarray
                local_data[colname] = np.array(col)  # делаем копию на всякий случай

        finally:
            h5.close()

        if all_data is None:
            # Первый файл — просто инициализируем
            all_data = local_data
        else:
            # Остальные — конкатенируем по оси 0
            for key, arr in local_data.items():
                if key not in all_data:
                    # Если вдруг новой колонки раньше не было — просто добавим
                    all_data[key] = arr
                    continue

                # Проверим совместимость размерностей (кроме оси 0)
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
        raise RuntimeError(f"No data loaded from pattern '{full_pattern}'")

    return all_data


def arrays_to_rows(data: Dict[str, np.ndarray]) -> Iterable[Dict[str, Any]]:
    """
    Вспомогательная функция: превращает набор numpy-массивов
    обратно в iterable dict-строк для save_to_hd5.

    Ожидаем, что все массивы имеют общую длину T по первой оси.
    """
    # найдём T по одному из ключей
    some_key = next(iter(data.keys()))
    T = data[some_key].shape[0]

    for i in range(T):
        row: Dict[str, Any] = {}
        for col in LUMITABLE_COLUMNS:
            arr = data[col]
            if arr.ndim == 1:
                row[col] = arr[i]
            else:
                row[col] = arr[i, ...]
        yield row
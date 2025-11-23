# src/hfcore/decorators.py
import time
import logging
from functools import wraps
from typing import Callable, Any, Iterable

from tqdm import tqdm

log = logging.getLogger("hfpipe")

def log_step(name: str | None = None):
    """
    Декоратор для логгирования начала/конца шага пайплайна.
    """
    def decorator(func: Callable):
        step_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.info(f"[{step_name}] started")
            result = func(*args, **kwargs)
            log.info(f"[{step_name}] finished")
            return result
        return wrapper
    return decorator


def timeit(name: str | None = None):
    """
    Декоратор для измерения времени выполнения функции.
    """
    def decorator(func: Callable):
        label = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            dt = time.time() - t0
            log.info(f"[{label}] took {dt:.2f} s")
            return result
        return wrapper
    return decorator


def with_progress(desc: str | None = None):
    """
    Декоратор для функций, которые возвращают iterable,
    и мы хотим обернуть его в tqdm.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Iterable[Any]:
            iterable = func(*args, **kwargs)
            return tqdm(iterable, desc=desc or func.__name__)
        return wrapper
    return decorator
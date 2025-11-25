# src/hfcore/decorators.py

"""
Utility decorators for logging, timing, and progress reporting.

These are used across the pipeline to ensure consistent logging and
to simplify instrumentation of long-running steps.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Iterable, Optional

from tqdm import tqdm

log = logging.getLogger("hfpipe")


# ----------------------------------------------------------------------
#  log_step
# ----------------------------------------------------------------------

def log_step(name: Optional[str] = None):
    """
    Decorator that logs the start and end of a pipeline step.

    Parameters
    ----------
    name : str, optional
        Human-readable step name. If omitted, the function name is used.

    Example
    -------
    @log_step("load_data")
    def load_data(...):
        ...
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


# ----------------------------------------------------------------------
#  timeit
# ----------------------------------------------------------------------

def timeit(name: Optional[str] = None):
    """
    Decorator that measures and logs execution time of the function.

    Parameters
    ----------
    name : str, optional
        Label used in the log output. Defaults to the function name.

    Example
    -------
    @timeit("compute_step1")
    def compute_step1(...):
        ...
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


# ----------------------------------------------------------------------
#  with_progress
# ----------------------------------------------------------------------

def with_progress(desc: Optional[str] = None):
    """
    Decorator for functions that return an iterable.
    Wraps the iterable into tqdm to show a progress bar.

    Important
    ---------
    - The decorated function must return an iterable.
    - The iterable is *not* consumed here — caller must iterate it.

    Parameters
    ----------
    desc : str, optional
        Description shown on the tqdm bar.

    Example
    -------
    @with_progress("processing LS")
    def generate_rows(...):
        for row in rows:
            yield row
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Iterable[Any]:
            iterable = func(*args, **kwargs)
            return tqdm(iterable, desc=desc or func.__name__)

        return wrapper

    return decorator
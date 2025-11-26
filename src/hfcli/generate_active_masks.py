#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import argparse
import logging
from typing import List, Optional

import numpy as np
import tables

from hfcore.hd5schema import BX_LEN

log = logging.getLogger("hfpipe.generate_masks")

# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

def autodetect_fills(beam_path: str) -> List[int]:
    """
    Detect all subdirectories of the form beam_path/<fill>/ where <fill> is an integer.

    Returns
    -------
    fills : list[int]
        Sorted list of discovered fill numbers.
    """
    pattern = os.path.join(beam_path, "*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    fills: List[int] = []
    for d in dirs:
        base = os.path.basename(d)
        try:
            fills.append(int(base))
        except ValueError:
            # Ignore non-numeric subdirectories
            continue

    fills.sort()
    return fills


def decode_status(value) -> str:
    """
    Decode a 'status' field from the beam table to str.

    Accepts bytes or str, always returns str.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def iter_beam_files_for_fill(beam_path: str, fill: int) -> List[str]:
    """
    Return a sorted list of *.hd5 files for a given fill under beam_path/<fill>/.

    If the directory does not exist, an empty list is returned.
    """
    fill_dir = os.path.join(beam_path, str(fill))
    if not os.path.isdir(fill_dir):
        return []

    pattern = os.path.join(fill_dir, "*.hd5")
    files = sorted(glob.glob(pattern))
    return files


def build_active_mask_for_fill(beam_path: str, fill: int) -> Optional[np.ndarray]:
    """
    Try to build an active BX mask (activeBXMask) for a given fill.

    Logic
    -----
    - Iterate over all files beam_path/<fill>/*.hd5
    - In each file, look for the /beam table
    - In /beam, search for the first row where:
        * row["fillnum"] == fill
        * row["status"] == "STABLE BEAMS"
    - Take row["collidable"] as the activeBXMask.

    If no suitable row is found in any file, returns None.
    """
    candidates = iter_beam_files_for_fill(beam_path, fill)
    if not candidates:
        log.warning("[fill %d] no beam files found under %s", fill, beam_path)
        return None

    log.info("[fill %d] beam candidates: %s", fill, candidates)

    for beam_file in candidates:
        log.info("[fill %d] opening beam file: %s", fill, beam_file)
        try:
            with tables.open_file(beam_file, mode="r") as h5:
                if not hasattr(h5.root, "beam"):
                    log.info("[fill %d] no /beam table in %s, skipping", fill, beam_file)
                    continue

                table: tables.Table = h5.root.beam
                log.info("[fill %d] using table '%s'", fill, table._v_pathname)

                selected_mask: Optional[np.ndarray] = None

                for row in table.iterrows():
                    row_fill = int(row["fillnum"])
                    if row_fill != fill:
                        continue

                    status_str = decode_status(row["status"])
                    if status_str != "STABLE BEAMS":
                        continue

                    coll = np.array(row["collidable"])
                    selected_mask = coll.astype(np.int32)
                    break

                if selected_mask is None:
                    log.info(
                        "[fill %d] no STABLE BEAMS rows with fillnum=%d in %s, trying next file",
                        fill,
                        fill,
                        beam_file,
                    )
                    continue

                if selected_mask.shape[0] != BX_LEN:
                    log.warning(
                        "[fill %d] collidable length=%d != BX_LEN=%d",
                        fill,
                        selected_mask.shape[0],
                        BX_LEN,
                    )

                return selected_mask

        except Exception as e:
            log.error(
                "[fill %d] error while reading %s: %s",
                fill,
                beam_file,
                e,
                exc_info=True,
            )
            # Try next file

    # If we get here, no suitable /beam row was found
    log.error(
        "[fill %d] no suitable /beam table with STABLE BEAMS found in any file",
        fill,
    )
    return None


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate activeBXMask_fill{fill}.npy from beam HD5 files."
    )
    parser.add_argument(
        "--beam-path",
        required=True,
        help=(
            "Base path to beam HD5 files, e.g. "
            "/eos/cms/store/group/dpg_bril/comm_bril/2025/physics/"
        ),
    )
    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Output directory for activeBXMask_fill{fill}.npy",
    )
    parser.add_argument(
        "--fills",
        nargs="+",
        type=int,
        help=(
            "List of fills to process (e.g. 9973 9974 9975). "
            "If omitted, fills are autodetected from subdirectories of --beam-path."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    os.makedirs(args.mask_dir, exist_ok=True)

    # If fills are not provided explicitly, autodetect from beam_path
    if args.fills is not None:
        fills = args.fills
    else:
        fills = autodetect_fills(args.beam_path)
        log.info("Autodetected fills: %s", fills)

    for fill in fills:
        try:
            log.info("[fill %d] start", fill)
            mask = build_active_mask_for_fill(
                beam_path=args.beam_path,
                fill=fill,
            )

            if mask is None:
                log.warning(
                    "[fill %d] mask not created (no beam/STABLE BEAMS), skipping",
                    fill,
                )
                continue

            out_path = os.path.join(args.mask_dir, f"activeBXMask_fill{fill}.npy")
            np.save(out_path, mask)
            log.info(
                "[fill %d] saved mask to %s (len=%d, n_active=%d)",
                fill,
                out_path,
                mask.shape[0],
                int(mask.sum()),
            )
        except Exception as e:
            log.error("[fill %d] FAILED: %s", fill, e, exc_info=True)


if __name__ == "__main__":
    main()
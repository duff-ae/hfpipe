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
    Build active BX mask for a fill by averaging all STABLE BEAMS collidable masks.

    Logic
    -----
    - Iterate over all files beam_path/<fill>/*.hd5
    - In each file, look for the /beam table
    - Collect all rows where:
        * row["fillnum"] == fill
        * row["status"] == "STABLE BEAMS"
    - Average collidable masks over all collected rows
    - Mark BX active if mean > 0.6
    - Force BX >= 3480 to inactive
    """
    candidates = iter_beam_files_for_fill(beam_path, fill)
    if not candidates:
        return None

    masks = []

    for beam_file in candidates:
        try:
            with tables.open_file(beam_file, mode="r") as h5:
                if not hasattr(h5.root, "beam"):
                    continue

                table: tables.Table = h5.root.beam

                if "collidable" not in table.colnames:
                    continue

                for row in table.iterrows():
                    row_fill = int(row["fillnum"])
                    if row_fill != fill:
                        continue

                    status_str = decode_status(row["status"]).strip().upper()
                    if status_str != "STABLE BEAMS":
                        continue

                    coll = np.asarray(row["collidable"], dtype=np.float32)
                    if coll.ndim != 1 or coll.size != BX_LEN:
                        continue

                    masks.append(coll)

        except Exception as e:
            log.error(
                "[fill %d] error while reading %s: %s",
                fill,
                beam_file,
                e,
                exc_info=True,
            )

    if not masks:
        return None

    mean_mask = np.mean(np.stack(masks, axis=0), axis=0)
    print([x for x in mean_mask])
    active_mask = (mean_mask > 0.2).astype(np.int32)

    # No real collidable BX should exist in the orbit tail
    active_mask[3480:] = 0

    return active_mask

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import tables as pt


SHORT_BASE = "/eos/cms/store/group/dpg_bril/comm_bril/2023/reprocessed/hfet/"
DONOR_BASE = "/eos/cms/store/group/dpg_bril/comm_bril/2023/online/per-bcid/"

SHORT_NODE = "/hfetlumi"
DONOR_NODE = "/hfetlumi"

KEY_COLUMNS = (
    "fillnum",
    "runnum",
    "lsnum",
    "nbnum",
)


def get_table(h5, path):
    node = h5.get_node(path)

    if not isinstance(node, pt.Table):
        raise RuntimeError(f"{path} is not a table")

    return node


def donor_files_for_pair(donor_base, fillnum, runnum):
    pattern = os.path.join(
        donor_base,
        str(fillnum),
        f"{fillnum}_{runnum}_*.hd5",
    )
    return sorted(glob.glob(pattern))


def build_donor_timestamp_map(short_arr, donor_base):
    """
    Build:

        (fill, run, ls, nb) -> timestampsec

    Runs without donor files are treated as interfill / junk runs.

    Duplicate donor keys are allowed only if timestampsec agrees.
    """

    pairs = sorted({
        (
            int(row["fillnum"]),
            int(row["runnum"]),
        )
        for row in short_arr
    })

    timestamp_map = {}
    valid_pairs = set()

    for fillnum, runnum in pairs:

        paths = donor_files_for_pair(
            donor_base,
            fillnum,
            runnum,
        )

        if not paths:
            print(
                f"   skip interfill run: "
                f"fill={fillnum}, run={runnum} "
                f"(no donor file)"
            )
            continue

        valid_pairs.add((fillnum, runnum))

        for path in paths:

            with pt.open_file(path, "r") as h5:
                tab = get_table(h5, DONOR_NODE)

                required = (*KEY_COLUMNS, "timestampsec")

                missing = [
                    name
                    for name in required
                    if name not in tab.colnames
                ]

                if missing:
                    raise RuntimeError(
                        f"Donor file {path} is missing columns: "
                        f"{missing}"
                    )

                for row in tab.read():

                    key = tuple(
                        int(row[name])
                        for name in KEY_COLUMNS
                    )

                    timestamp = int(
                        row["timestampsec"]
                    )

                    previous = timestamp_map.get(key)

                    if previous is None:
                        timestamp_map[key] = timestamp

                    elif previous != timestamp:
                        raise RuntimeError(
                            f"Conflicting timestampsec for donor key "
                            f"{key}: {previous} vs {timestamp}"
                        )

    return timestamp_map, valid_pairs


def fix_legacy_nbnum(arr):
    """
    Fix only the known legacy nibble convention:

        3, 7, 11, ..., 63
            ->
        4, 8, 12, ..., 64

    Already-correct values are untouched.

    Any other strange nbnum values are left unchanged here.
    They will later fail donor matching and be removed.
    """

    out = arr.copy()

    nb = out["nbnum"].astype(np.int64)

    correct = (
        (nb >= 4)
        & (nb <= 64)
        & (nb % 4 == 0)
    )

    legacy = (
        (nb >= 3)
        & (nb <= 63)
        & (nb % 4 == 3)
    )

    other = ~(correct | legacy)

    n_correct = int(np.count_nonzero(correct))
    n_legacy = int(np.count_nonzero(legacy))
    n_other = int(np.count_nonzero(other))

    print(
        f"   nbnum check: "
        f"{n_correct} already correct, "
        f"{n_legacy} legacy, "
        f"{n_other} other"
    )

    if n_legacy:
        out["nbnum"][legacy] += 1

        print(
            f"   fixed legacy nbnum: "
            f"{n_legacy} row(s)"
        )

    if n_other:
        print(
            f"   leaving {n_other} unusual nbnum row(s) unchanged; "
            f"they will be filtered by donor matching"
        )

    return out


def safe_replace_table(
    path,
    out,
    attrs,
    filters,
    title,
):
    """
    Safely replace /hfetlumi.

    Procedure:

        /hfetlumi       -> /hfetlumi__old
        /hfetlumi__new  -> /hfetlumi

    Only after the new node is successfully in place is
    /hfetlumi__old removed.

    If the second rename fails, the old table is restored.
    """

    node_name = "hfetlumi"
    tmp_name = "hfetlumi__new"
    backup_name = "hfetlumi__old"

    with pt.open_file(path, "r+") as h5:

        children = h5.root._v_children

        # --------------------------------------------------------
        # Recover from an interrupted previous swap if necessary.
        # --------------------------------------------------------

        if node_name not in children:

            if backup_name in children:
                print(
                    f"   recovering stale /{backup_name} "
                    f"to /{node_name}"
                )

                h5.rename_node(
                    "/",
                    node_name,
                    name=backup_name,
                )

                h5.flush()

            elif tmp_name in children:
                raise RuntimeError(
                    f"/{node_name} is missing but /{tmp_name} exists. "
                    f"Refusing to overwrite an unresolved temporary table."
                )

            else:
                raise RuntimeError(
                    f"/{node_name} is missing and no recovery node exists"
                )

        # Refresh after possible recovery.
        children = h5.root._v_children

        # --------------------------------------------------------
        # Remove stale temporary nodes only while canonical table
        # is safely present.
        # --------------------------------------------------------

        if tmp_name in children:
            print(
                f"   removing stale /{tmp_name}"
            )
            h5.remove_node("/", tmp_name)

        children = h5.root._v_children

        if backup_name in children:
            print(
                f"   removing stale /{backup_name}"
            )
            h5.remove_node("/", backup_name)

        # --------------------------------------------------------
        # Create complete replacement table.
        # --------------------------------------------------------

        new_tab = h5.create_table(
            "/",
            tmp_name,
            obj=out,
            title=title,
            filters=filters,
        )

        for name, value in attrs.items():
            try:
                setattr(
                    new_tab.attrs,
                    name,
                    value,
                )
            except Exception as exc:
                print(
                    f"   WARNING: failed to restore "
                    f"attribute {name}: {exc}"
                )

        new_tab.flush()
        h5.flush()

        # Basic sanity before touching the old table.
        if new_tab.nrows != len(out):
            raise RuntimeError(
                f"Temporary table has {new_tab.nrows} rows, "
                f"expected {len(out)}"
            )

        # --------------------------------------------------------
        # Safe swap.
        # --------------------------------------------------------

        print(
            f"   swapping /{node_name} safely"
        )

        # old -> backup
        h5.rename_node(
            "/",
            backup_name,
            name=node_name,
        )

        h5.flush()

        try:

            # new -> canonical
            h5.rename_node(
                "/",
                node_name,
                name=tmp_name,
            )

            h5.flush()

        except Exception:

            print(
                "   ERROR during table swap; "
                "restoring original /hfetlumi"
            )

            children = h5.root._v_children

            # Restore old table if canonical name was not created.
            if (
                node_name not in children
                and backup_name in children
            ):
                h5.rename_node(
                    "/",
                    node_name,
                    name=backup_name,
                )
                h5.flush()

            raise

        # --------------------------------------------------------
        # Verify canonical table BEFORE deleting backup.
        # --------------------------------------------------------

        final_tab = get_table(
            h5,
            SHORT_NODE,
        )

        if final_tab.nrows != len(out):

            print(
                "   ERROR: new /hfetlumi failed row-count check; "
                "rolling back"
            )

            h5.remove_node(
                "/",
                node_name,
            )

            h5.rename_node(
                "/",
                node_name,
                name=backup_name,
            )

            h5.flush()

            raise RuntimeError(
                f"Final table has {final_tab.nrows} rows, "
                f"expected {len(out)}"
            )

        # --------------------------------------------------------
        # New table is valid. Only now remove backup.
        # --------------------------------------------------------

        h5.remove_node(
            "/",
            backup_name,
        )

        h5.flush()


def process_file(path, donor_base, dry_run=False):

    print()
    print("=" * 80)
    print(f"-- {path}")

    # ------------------------------------------------------------
    # Read original reprocessed table.
    # ------------------------------------------------------------

    with pt.open_file(path, "r") as h5:

        tab = get_table(
            h5,
            SHORT_NODE,
        )

        short_arr = tab.read()

        short_attrs = {
            name: getattr(tab.attrs, name)
            for name in tab.attrs._f_list()
        }

        short_filters = tab.filters
        short_title = tab._v_title

    if len(short_arr) == 0:
        print("   empty table, skip")
        return

    original_rows = len(short_arr)

    required = (
        *KEY_COLUMNS,
        "timestampsec",
    )

    missing = [
        name
        for name in required
        if name not in short_arr.dtype.names
    ]

    if missing:
        raise RuntimeError(
            f"Short table is missing columns: {missing}"
        )

    # ------------------------------------------------------------
    # Load donor timestampsec.
    # ------------------------------------------------------------

    timestamp_map, valid_pairs = build_donor_timestamp_map(
        short_arr,
        donor_base,
    )

    if not timestamp_map:
        raise RuntimeError(
            "No donor rows loaded"
        )

    # ------------------------------------------------------------
    # Remove runs completely absent from donor.
    # ------------------------------------------------------------

    keep_run = np.array(
        [
            (
                int(row["fillnum"]),
                int(row["runnum"]),
            ) in valid_pairs
            for row in short_arr
        ],
        dtype=bool,
    )

    n_removed_runs = int(
        np.count_nonzero(~keep_run)
    )

    if n_removed_runs:
        print(
            f"   removing {n_removed_runs}/{len(short_arr)} "
            f"row(s) from runs absent in donor files"
        )

    short_arr = short_arr[keep_run]

    if len(short_arr) == 0:
        raise RuntimeError(
            "No rows left after removing runs absent from donor"
        )

    # ------------------------------------------------------------
    # Fix only legacy nibble numbers.
    # ------------------------------------------------------------

    short_arr = fix_legacy_nbnum(
        short_arr
    )

    # ------------------------------------------------------------
    # Match against donor.
    #
    # Rows absent from donor are discarded.
    # ------------------------------------------------------------

    matched_mask = np.zeros(
        len(short_arr),
        dtype=bool,
    )

    donor_timestamps = np.zeros(
        len(short_arr),
        dtype=np.uint32,
    )

    unmatched = []

    for i, row in enumerate(short_arr):

        key = tuple(
            int(row[name])
            for name in KEY_COLUMNS
        )

        timestamp = timestamp_map.get(
            key
        )

        if timestamp is None:
            unmatched.append(key)
            continue

        matched_mask[i] = True
        donor_timestamps[i] = timestamp

    if unmatched:
        print(
            f"   removing {len(unmatched)}/{len(short_arr)} "
            f"row(s) with no donor match"
        )

        print(
            f"   first unmatched keys: "
            f"{unmatched[:20]}"
        )

    short_arr = short_arr[
        matched_mask
    ]

    donor_timestamps = donor_timestamps[
        matched_mask
    ]

    if len(short_arr) == 0:
        raise RuntimeError(
            "No rows left after donor matching"
        )

    print(
        f"   matched rows kept: "
        f"{len(short_arr)}"
    )

    # ------------------------------------------------------------
    # timestampsec diagnostic.
    # ------------------------------------------------------------

    old_timestamp = short_arr[
        "timestampsec"
    ].astype(np.int64)

    new_timestamp = donor_timestamps.astype(
        np.int64
    )

    different = (
        old_timestamp != new_timestamp
    )

    n_different = int(
        np.count_nonzero(different)
    )

    n_zero = int(
        np.count_nonzero(
            old_timestamp == 0
        )
    )

    print()
    print("   timestampsec check:")

    print(
        f"      different: "
        f"{n_different}/{len(short_arr)}"
    )

    print(
        f"      zero in short: "
        f"{n_zero}/{len(short_arr)}"
    )

    if n_different:

        print(
            "      first differences:"
        )

        for i in np.flatnonzero(
            different
        )[:10]:

            row = short_arr[i]

            key = tuple(
                int(row[name])
                for name in KEY_COLUMNS
            )

            print(
                f"         {key}: "
                f"short={old_timestamp[i]}, "
                f"donor={new_timestamp[i]}"
            )

    # ------------------------------------------------------------
    # Modify ONLY timestampsec.
    # ------------------------------------------------------------

    out = short_arr.copy()

    out["timestampsec"] = donor_timestamps.astype(
        out["timestampsec"].dtype,
        copy=False,
    )

    print()
    print(
        f"   timestampsec replaced: "
        f"{n_different}/{len(out)} row(s)"
    )

    print(
        f"   final rows: {len(out)} "
        f"(input {original_rows}, "
        f"removed {original_rows - len(out)})"
    )

    # ------------------------------------------------------------
    # Dry run.
    # ------------------------------------------------------------

    if dry_run:
        print(
            "   DRY RUN: file not modified"
        )
        return

    # ------------------------------------------------------------
    # Safe write.
    # ------------------------------------------------------------

    safe_replace_table(
        path=path,
        out=out,
        attrs=short_attrs,
        filters=short_filters,
        title=short_title,
    )

    print(
        f"   DONE: wrote {len(out)} rows"
    )


def files_for_fill(fillnum):

    path = os.path.join(
        SHORT_BASE,
        str(fillnum),
        f"{fillnum}.hd5",
    )

    return [path] if os.path.isfile(path) else []


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Repair special reprocessed HFET files: "
            "remove rows absent from original donor data, "
            "fix legacy nbnum numbering, and restore timestampsec. "
            "All other columns remain unchanged."
        )
    )

    parser.add_argument(
        "--file",
        help="Process one HD5 file",
    )

    parser.add_argument(
        "--fill",
        type=int,
        help="Process one fill",
    )

    parser.add_argument(
        "--fill-min",
        type=int,
        help="First fill to process, inclusive",
    )

    parser.add_argument(
        "--fill-max",
        type=int,
        help="Last fill to process, inclusive",
    )

    parser.add_argument(
        "--donor-base",
        default=DONOR_BASE,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run all checks and show proposed changes "
            "without modifying files"
        ),
    )

    args = parser.parse_args()

    if args.file:

        paths = [
            args.file
        ]

    elif args.fill is not None:

        paths = files_for_fill(
            args.fill
        )

    else:

        paths = []

        base = Path(
            SHORT_BASE
        )

        for fill_dir in sorted(
            (
                p
                for p in base.iterdir()
                if p.is_dir() and p.name.isdigit()
            ),
            key=lambda p: int(p.name),
        ):

            fillnum = int(
                fill_dir.name
            )

            if (
                args.fill_min is not None
                and fillnum < args.fill_min
            ):
                continue

            if (
                args.fill_max is not None
                and fillnum > args.fill_max
            ):
                continue

            paths.extend(
                files_for_fill(fillnum)
            )

    if not paths:
        raise SystemExit(
            "No input files found"
        )

    for path in paths:

        process_file(
            path,
            args.donor_base,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
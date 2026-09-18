#!/usr/bin/env python3

import argparse
import glob
import os

import tables as pt


# python3 file_check.py --fill 8775

SHORT_BASE = "/eos/cms/store/group/dpg_bril/comm_bril/2023/reprocessed/hfet/"
DONOR_BASE = "/eos/cms/store/group/dpg_bril/comm_bril/2023/online/per-bcid/"

NODE = "/hfetlumi"
KEY_COLUMNS = ("fillnum", "runnum", "lsnum", "nbnum")


def key(row):
    return tuple(int(row[x]) for x in KEY_COLUMNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill", type=int, required=True)
    args = parser.parse_args()

    fill = args.fill

    short_path = os.path.join(
        SHORT_BASE,
        str(fill),
        f"{fill}.hd5",
    )

    with pt.open_file(short_path, "r") as h5:
        short = h5.get_node(NODE).read()

    print(f"SHORT rows: {len(short)}")

    short_by_run = {}

    for row in short:
        run = int(row["runnum"])
        short_by_run.setdefault(run, set()).add(key(row))

    total_matched = 0
    total_short = 0

    for run, short_keys in sorted(short_by_run.items()):

        pattern = os.path.join(
            DONOR_BASE,
            str(fill),
            f"{fill}_{run}_*.hd5",
        )

        donor_files = sorted(glob.glob(pattern))

        print()
        print(f"RUN {run}")
        print(f"  short keys: {len(short_keys)}")
        print(f"  donor files: {len(donor_files)}")

        if not donor_files:
            print("  NO DONOR FILE")
            continue

        donor_keys = set()

        for path in donor_files:
            with pt.open_file(path, "r") as h5:
                arr = h5.get_node(NODE).read()

            donor_keys.update(key(row) for row in arr)

        matched = short_keys & donor_keys
        only_short = short_keys - donor_keys
        only_donor = donor_keys - short_keys

        print(f"  donor keys: {len(donor_keys)}")
        print(f"  matched:    {len(matched)}")
        print(f"  short-only: {len(only_short)}")
        print(f"  donor-only: {len(only_donor)}")

        if only_short:
            print("  first short-only keys:")
            for x in list(sorted(only_short))[:5]:
                print(f"    {x}")

        total_short += len(short_keys)
        total_matched += len(matched)

    print()
    print("=" * 60)
    print(f"TOTAL SHORT:   {total_short}")
    print(f"TOTAL MATCHED: {total_matched}")
    print(f"FRACTION:      {total_matched / total_short:.6f}")


if __name__ == "__main__":
    main()
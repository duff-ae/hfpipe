#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump and compare the (fillnum, runnum, lsnum, nbnum) keys from a short file
and its matching donor file(s), to see exactly how they diverge instead of
guessing from match/mismatch counts alone.

Usage:
    python3 compare_keys.py <short.hd5> "<donor_glob>"

Example:
    python3 compare_keys.py \\
        /cephfs/brilshare/alshevel/l1scouting_muon/9996/9996_384374.hd5 \\
        "/eos/cms/store/group/dpg_bril/comm_bril/2024/online/per-bcid/9996/*.hd5"
"""
import sys
import glob
import numpy as np
import tables as pt

SHORT_NODE = "/l1scoutlumi"
DONOR_NODE = "/hfetlumi"
KEYS = ("fillnum", "runnum", "lsnum", "nbnum")


def dump_keys(path, node):
    with pt.open_file(path, "r") as h5:
        tab = h5.get_node(node)
        arr = tab.read()
    return np.array([[int(r[k]) for k in KEYS] for r in arr])


def summarize(name, keys):
    print(f"--- {name}: {len(keys)} row(s) ---")
    if len(keys) == 0:
        return
    runs = sorted(set(keys[:, 1].tolist()))
    print(f"  run(s): {runs}")
    for run in runs:
        sub = keys[keys[:, 1] == run]
        ls, nb = sub[:, 2], sub[:, 3]
        order = np.lexsort((nb, ls))
        pairs = list(zip(ls[order].tolist(), nb[order].tolist()))

        uniqueNb = sorted(set(nb.tolist()))
        nbSteps = sorted(set(np.diff(np.array(uniqueNb))[np.diff(np.array(uniqueNb)) > 0].tolist())) \
            if len(uniqueNb) > 1 else []

        print(f"  run {run}: {len(sub)} rows")
        print(f"    lsnum range: [{ls.min()}, {ls.max()}]")
        print(f"    nbnum range: [{nb.min()}, {nb.max()}], "
              f"distinct nbnum values (first 20): {uniqueNb[:20]}")
        print(f"    step(s) seen between consecutive distinct nbnum values: {nbSteps}")
        print(f"    first 10 (ls, nb) pairs: {pairs[:10]}")
        print(f"    last 10 (ls, nb) pairs:  {pairs[-10:]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <short.hd5> <donor_glob>")

    short_path, donor_glob = sys.argv[1], sys.argv[2]

    short_keys = dump_keys(short_path, SHORT_NODE)
    summarize(f"SHORT ({short_path})", short_keys)

    donor_paths = sorted(glob.glob(donor_glob))
    print(f"\nFound {len(donor_paths)} donor file(s) matching {donor_glob!r}")
    all_donor_keys = []
    for dp in donor_paths:
        try:
            all_donor_keys.append(dump_keys(dp, DONOR_NODE))
        except Exception as e:
            print(f"  skip {dp}: {e}")

    if all_donor_keys:
        donor_keys = np.concatenate(all_donor_keys, axis=0)
        print()
        summarize("DONOR (all files combined)", donor_keys)

        # direct overlap check
        short_set = {tuple(r) for r in short_keys.tolist()}
        donor_set = {tuple(r) for r in donor_keys.tolist()}
        print(f"\nExact-key overlap: {len(short_set & donor_set)} / {len(short_set)} "
              f"short rows have a matching donor row.")

        # same but ignoring nbnum, to isolate whether lsnum alone lines up
        short_ls = {(r[0], r[1], r[2]) for r in short_keys.tolist()}
        donor_ls = {(r[0], r[1], r[2]) for r in donor_keys.tolist()}
        print(f"(fill,run,ls)-only overlap (ignoring nbnum): {len(short_ls & donor_ls)} / "
              f"{len(short_ls)} distinct short (fill,run,ls) triples have a matching donor one.")

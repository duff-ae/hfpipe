#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import numpy as np
import tables as pt
from pathlib import Path

SHORT_BASE = "/cephfs/brilshare/alshevel/l1scouting_muon2"
DONOR_BASE = "/eos/cms/store/group/dpg_bril/comm_bril/2024/online/per-bcid/"

# Donor and short tables are NOT the same node name -- donor (the official
# online per-bcid files) uses "hfetlumi" regardless of which detector's
# scouting data it's paired with; our own converted files use "l1scoutlumi".
DONOR_NODE = "/hfetlumi"
SHORT_NODE = "/l1scoutlumi"

# Donor filenames are "<fill>_<run>_<start>_<end>.hd5" -- <run> is matched
# as an exact underscore-bounded numeric field, not a loose substring, so a
# run number that happens to appear inside another field (fill, start,
# end) can't cause a false match.
DONOR_FILENAME_RE_TEMPLATE = r'^\d+_{run}_\d+_\d+\.hd5$'

# Old conversions (before the wrap fix landed in scdaq2hd5.py) stored nbnum
# as a raw, monotonically growing per-run counter (0, 4, 8, ..., never
# wrapping) instead of cycling every LS like the real hardware/donor
# convention: confirmed empirically, donor nbnum always cycles in
# [NIBBLE_MULTIPLIER, NIBBLES_PER_LS*NIBBLE_MULTIPLIER] (i.e. [4, 64]) and
# resets to NIBBLE_MULTIPLIER at the start of every new LS.
NIBBLES_PER_LS = 16
NIBBLE_MULTIPLIER = 4

# Fixed active-BX mask: everything below the abort gap, except one
# permanently-dirty calibration-pulse-like bin. No data-driven derivation.
ABORT_GAP_START_BX = 3480   # BX >= this is always excluded (LHC abort gap)
DIRTY_BIN_BX = 3527         # always-dirty calibration bin, always excluded too

# Calibration: avg = avgraw * FREQ_REV_HZ / sigvis. avgraw/avg are SUMS over
# active BX, not means -- historical convention, kept as-is.
FREQ_REV_HZ = 11245.6
DEFAULT_SIGVIS = 264000.0


def fix_ls_nbnum(arr, multiplier=NIBBLE_MULTIPLIER, nibbles_per_ls=NIBBLES_PER_LS):
    """Repair lsnum/nbnum, per run, without re-running the whole conversion.
    Three cases, detected from the data itself:

    (1) OLD un-wrapped conversion (before the wrap fix landed in
        scdaq2hd5.py): nbnum is a raw, monotonically-growing-past-one-LS
        counter (max > period). Unwrap it: cycle = old_nbnum // period,
        normalized so the run's own first row is cycle 0; new lsnum =
        cycle + 1; new nbnum = (old_nbnum % period) + multiplier.

    (2) Wrapped, but with the WRONG multiplier (scdaq2hd5.py's --nibble
        defaults to 1, not the real hardware value of 4 that the donor
        convention uses): max(nbnum) <= nibbles_per_ls (16), i.e. nbnum
        cycles in [1, 16] instead of [4, 64]. This is losslessly fixed by
        scaling by exactly `multiplier`: because
        nbnum_m = m*((raw_counter % 16)) + m = m * nbnum_1 for any m, a
        run converted with multiplier=1 has nbnum_1 stored, and
        nbnum_correct = multiplier * nbnum_1 exactly recovers what
        multiplier=4 would have produced -- no information was lost by the
        wrong CLI default, just the wrong scale.

    (3) Already correctly wrapped with the real multiplier (values already
        cycle in [multiplier, period]): left as-is.

    Returns a NEW array; does not mutate the input. Rows are matched back
    by their original position, so this is safe regardless of row order.
    """
    period = nibbles_per_ls * multiplier
    out = arr.copy()
    for run in np.unique(arr['runnum']):
        idx = np.where(arr['runnum'] == run)[0]
        oldNb = arr['nbnum'][idx].astype(np.int64)
        m = int(oldNb.max())

        if m > period:
            # Case (1): old un-wrapped raw counter.
            cycle = oldNb // period
            cycle = cycle - cycle.min()
            out['lsnum'][idx] = (cycle + 1).astype(out['lsnum'].dtype)
            out['nbnum'][idx] = ((oldNb % period) + multiplier).astype(out['nbnum'].dtype)
            print(f"     run {run}: nbnum looks like an OLD un-wrapped counter (max={m} > "
                  f"period={period}) -- unwrapped, ls range [1, {int(cycle.max()) + 1}]")

        elif np.all(oldNb % multiplier == 0):
            # Case (3): already correct -- every value is a multiple of the
            # real multiplier, as the correct formula always produces.
            print(f"     run {run}: nbnum already within [.., {period}] (max={m}), all values "
                  f"multiples of {multiplier} -- looks already correctly wrapped, leaving as-is")

        else:
            # Case (2): wrapped, but with the wrong (default=1) multiplier
            # -- values aren't all multiples of the real multiplier.
            out['nbnum'][idx] = (oldNb * multiplier).astype(out['nbnum'].dtype)
            print(f"     run {run}: nbnum looks wrapped with the WRONG multiplier "
                  f"(max={m}, not all multiples of {multiplier} -- i.e. converted with "
                  f"--nibble 1 instead of {multiplier}) -- rescaled *{multiplier} (lossless), "
                  f"lsnum left as-is")
    return out


def get_table(h5, path):
    node = h5.get_node(path)
    if not isinstance(node, pt.Table):
        raise RuntimeError(f"{path} is not a table")
    return node


def build_run_fills_index(donorBase, verbose=True):
    """ONE single pass over the whole donor tree: for every donor file
    <fill>/<fill>_<run>_<start>_<end>.hd5, record that `run` appears under
    `fill`. Returns {run: set(fills)}.

    This replaces re-scanning every fill directory for every run (which is
    O(runs * fills) and exactly the kind of thing that gets unusably slow
    over a whole donor tree) with a single O(total donor files) pass,
    reused for every run afterwards. Runs whose set has more than one fill
    are the confirmed multi-fill ("boundary") runs -- printed here, up
    front, before any merging happens."""
    index = {}
    base = Path(donorBase)
    if not base.is_dir():
        print(f"[WARN] donor base not found or not accessible: {donorBase}")
        return index
    pattern = re.compile(r'^\d+_(\d+)_\d+_\d+\.hd5$')
    for fillDir in sorted(p for p in base.iterdir() if p.is_dir()):
        try:
            fillNumber = int(fillDir.name)
        except ValueError:
            continue
        for f in fillDir.iterdir():
            if not f.is_file():
                continue
            m = pattern.match(f.name)
            if not m:
                continue
            runNumber = int(m.group(1))
            index.setdefault(runNumber, set()).add(fillNumber)

    if verbose:
        multiFill = {r: fs for r, fs in index.items() if len(fs) > 1}
        print(f"Run->fill index: {len(index)} run(s) seen across {donorBase}, "
              f"{len(multiFill)} of them span more than one fill")
        for r in sorted(multiFill):
            print(f"  [MULTI-FILL-RUN] run={r}: fills={sorted(multiFill[r])}")
    return index


def find_donor_paths_for_candidates(donorBase, run, candidateFills):
    """Only look inside the (usually 1, sometimes 2-3) fill directories
    already known -- from the run->fills index -- to contain this run's
    donor files. No blind scanning of the whole tree."""
    pattern = re.compile(DONOR_FILENAME_RE_TEMPLATE.format(run=run))
    matches = []
    for fillNumber in candidateFills:
        fillDir = Path(donorBase) / str(fillNumber)
        if not fillDir.is_dir():
            continue
        for f in fillDir.iterdir():
            if f.is_file() and pattern.match(f.name):
                matches.append(f)
    return sorted(matches)


def load_donor_map_by_run(donorPaths, verbose=True):
    """Merge every matched donor file's rows into one dict keyed by
    (runnum, lsnum, nbnum) -- fillnum is deliberately NOT part of the key,
    since resolving fillnum per row is the whole point. If the same
    (run, ls, nb) shows up with two different fillnum values across donor
    files (shouldn't normally happen), the first one found is kept and a
    warning is printed."""
    donor_map = {}
    donor_dtype = None
    donor_attrs = {}
    conflicts = 0
    for dp in donorPaths:
        with pt.open_file(dp, "r") as h5d:
            try:
                donor_tab = get_table(h5d, DONOR_NODE)
            except Exception:
                continue
            if donor_dtype is None:
                donor_dtype = donor_tab.dtype
            try:
                for k in donor_tab.attrs._f_list():
                    donor_attrs[k] = getattr(donor_tab.attrs, k)
            except Exception:
                pass
            for row in donor_tab.read():
                key = (int(row['runnum']), int(row['lsnum']), int(row['nbnum']))
                if key in donor_map and int(donor_map[key]['fillnum']) != int(row['fillnum']):
                    conflicts += 1
                    continue
                donor_map[key] = row
    if verbose and conflicts:
        print(f"     [WARN] {conflicts} (run,ls,nb) key(s) had conflicting fillnum across "
              f"donor files; kept the first one found")
    return donor_map, donor_dtype, donor_attrs


def build_fixed_mask(nbx, abort_gap_start=ABORT_GAP_START_BX, dirty_bin=DIRTY_BIN_BX):
    """Fixed active-BX mask: True for BX < abort_gap_start, except
    dirty_bin (a permanently-dirty calibration-pulse-like bin), which is
    always excluded too. No data dependence -- same mask for every row."""
    mask = np.zeros(nbx, dtype=bool)
    mask[:abort_gap_start] = True
    if 0 <= dirty_bin < nbx:
        mask[dirty_bin] = False
    return mask


def recompute_avgraw_avg(bxraw_row, mask, freq_rev_hz, sigvis):
    """avgraw = SUM of bxraw over the active-BX mask (historical convention
    despite the name -- NOT a mean). avg is the same sum calibrated to a
    rate: avg = avgraw * freq_rev_hz / sigvis."""
    n = min(len(bxraw_row), len(mask))
    active = np.asarray(bxraw_row[:n])[mask[:n]]
    avgraw = float(active.sum()) if active.size else 0.0
    return avgraw, avgraw * freq_rev_hz / sigvis





def build_output_dtype(nbx, donor_dtype):
    """Minimal output schema: our own scalar/array columns, plus ONLY
    timestampsec/timestampmsec pulled in from donor -- nothing else from
    donor is merged in anymore (totsize, publishnnb, datasourceid, algoid,
    channelid, payloadtype, calibtag, maskhigh, masklow: not merged, not
    written out at all)."""
    def donorFieldDtype(name, fallback):
        if donor_dtype is not None and name in donor_dtype.names:
            return donor_dtype.fields[name][0]
        return np.dtype(fallback)

    return np.dtype([
        ('fillnum', 'u4'), ('runnum', 'u4'), ('lsnum', 'u4'), ('nbnum', 'u4'),
        ('timestampsec', donorFieldDtype('timestampsec', 'u4')),
        ('timestampmsec', donorFieldDtype('timestampmsec', 'u4')),
        ('avgraw', 'u4'), ('avg', 'f8'),
        ('bxraw', 'u2', (nbx,)), ('bx', 'f8', (nbx,)),
    ])


def compute_calibrated_bx(bxraw_row, freq_rev_hz, sigvis):
    """bx = bxraw calibrated to a rate PER BIN (no mask, no summing --
    every one of the 3564 bins individually): bx = bxraw * freq_rev_hz /
    sigvis. This is the per-bin counterpart to avg (which sums first, over
    the active-BX mask, then calibrates)."""
    return np.asarray(bxraw_row, dtype=np.float64) * freq_rev_hz / sigvis


def process_short_file(short_path, donor_base, runFillsIndex,
                        abort_gap_start=ABORT_GAP_START_BX,
                        dirty_bin=DIRTY_BIN_BX,
                        sigvis=DEFAULT_SIGVIS, dry_run=False):
    """STEP 1: resolve real fillnum per row from donor data -- looked up
    only in the fill directories the run->fills index already says this
    run appears under -- and merge donor columns in. STEP 2: recompute
    avgraw/avg with the fixed BX<abort_gap_start & !=dirty_bin mask.
    Rewrites the file in place (unless dry_run)."""
    print(f"  -- {short_path}")
    mode = "r" if dry_run else "r+"
    with pt.open_file(short_path, mode) as h5s:
        short_tab = get_table(h5s, SHORT_NODE)
        short_arr = short_tab.read()

        print(f"     short table schema: {short_arr.dtype.names}")

        short_arr = fix_ls_nbnum(short_arr)
        order = np.lexsort((short_arr['nbnum'], short_arr['lsnum']))
        short_arr = short_arr[order]

        runnum = int(short_arr['runnum'][0]) if len(short_arr) else None
        if runnum is None:
            print("     [SKIP] empty file")
            return

        # ---------------- STEP 1: donor-based fill resolution ----------------
        candidateFills = runFillsIndex.get(runnum)
        if not candidateFills:
            print(f"     [SKIP] run {runnum} not found in the run->fill index "
                  f"(no donor file anywhere under {donor_base})")
            return
        if len(candidateFills) > 1:
            print(f"     run {runnum} is a known MULTI-FILL run, candidates: "
                  f"{sorted(candidateFills)}")
        donorPaths = find_donor_paths_for_candidates(donor_base, runnum, candidateFills)
        if not donorPaths:
            print(f"     [SKIP] run {runnum}: index said fill(s) {sorted(candidateFills)} "
                  f"but no matching donor file found there")
            return
        donor_map, donor_dtype, donor_attrs = load_donor_map_by_run(donorPaths)
        if donor_dtype is None:
            print(f"     [SKIP] no donor {DONOR_NODE} table found for run {runnum}")
            return

        if 'bxraw' not in short_arr.dtype.names or len(short_arr) == 0:
            print("     [SKIP] no bxraw column or no rows; nothing to compute")
            return

        short_cols = set(short_arr.dtype.names)
        missing = [c for c in donor_dtype.names if c not in short_cols]
        print(f"     donor has {len(missing)} extra column(s) not in short: {missing} -- "
              f"only timestampsec/timestampmsec of these are actually merged in; the rest "
              f"(totsize, publishnnb, datasourceid, algoid, channelid, payloadtype, calibtag, "
              f"maskhigh, masklow) are ignored")
        print(f"     donor files found for run {runnum}: {len(donorPaths)} file(s) across "
              f"candidate fill(s) {sorted(candidateFills)}")

        # ---- diagnostics: where exactly does short vs. donor coverage diverge? ----
        shortLs, shortNb = short_arr['lsnum'], short_arr['nbnum']
        print(f"     short file own coverage: ls range=[{int(shortLs.min())}-{int(shortLs.max())}], "
              f"nb range=[{int(shortNb.min())}-{int(shortNb.max())}], "
              f"distinct ls={len(np.unique(shortLs))}")
        donorByFill = {}
        for (r, ls, nb), drow in donor_map.items():
            f = int(drow['fillnum'])
            bucket = donorByFill.setdefault(f, {'ls': [], 'nb': []})
            bucket['ls'].append(ls)
            bucket['nb'].append(nb)
        for f in sorted(donorByFill):
            lsArr, nbArr = donorByFill[f]['ls'], donorByFill[f]['nb']
            print(f"     donor fill={f}: rows={len(lsArr)}, ls range=[{min(lsArr)}-{max(lsArr)}], "
                  f"nb range=[{min(nbArr)}-{max(nbArr)}], distinct ls={len(set(lsArr))}")

        nbx = short_arr['bxraw'].shape[1]
        outDtype = build_output_dtype(nbx, donor_dtype)
        out = np.zeros(len(short_arr), dtype=outDtype)
        matched = 0
        unmatched = 0
        for i, srow in enumerate(short_arr):
            key = (int(srow['runnum']), int(srow['lsnum']), int(srow['nbnum']))
            drow = donor_map.get(key)
            out[i]['runnum'] = srow['runnum']
            out[i]['lsnum'] = srow['lsnum']
            out[i]['nbnum'] = srow['nbnum']
            out[i]['bxraw'] = srow['bxraw']
            if drow is not None:
                out[i]['fillnum'] = drow['fillnum']       # authoritative, from donor
                out[i]['timestampsec'] = drow['timestampsec']
                out[i]['timestampmsec'] = drow['timestampmsec']
                matched += 1
            else:
                out[i]['fillnum'] = srow['fillnum']        # unverified fallback
                out[i]['timestampsec'] = 0
                out[i]['timestampmsec'] = 0
                unmatched += 1

        if unmatched:
            print(f"     [WARN] {unmatched}/{len(short_arr)} row(s) had no donor match for "
                  f"(run,ls,nb); their fillnum is UNVERIFIED (kept from the original heuristic "
                  f"assignment) and timestampsec/timestampmsec are 0 (no donor row to take "
                  f"them from)")

        resolvedFills, counts = np.unique(out['fillnum'], return_counts=True)
        if len(resolvedFills) > 1:
            details = []
            for f in resolvedFills:
                idx = np.where(out['fillnum'] == f)[0]
                lsMin = int(out['lsnum'][idx].min())
                lsMax = int(out['lsnum'][idx].max())
                details.append(f"fill={int(f)} rows={idx.size} ls=[{lsMin}-{lsMax}]")
            print(f"     [FILL-SPLIT] run={runnum}: rows resolved to {len(resolvedFills)} "
                  f"distinct fill(s) via donor (run,ls,nb) matching -- confirmed fill "
                  f"boundary: " + ", ".join(details))
        else:
            print(f"     run={runnum}: all matched rows resolve to a single fill "
                  f"({int(resolvedFills[0]) if len(resolvedFills) else 'n/a'})")

        # ---------------- STEP 2: compute avgraw/avg/bx from our own bxraw ----------------
        recomputed = 0
        fixedMask = build_fixed_mask(nbx, abort_gap_start, dirty_bin)
        bxrawAll = short_arr['bxraw']
        for i in range(len(short_arr)):
            avgraw, avg = recompute_avgraw_avg(bxrawAll[i], fixedMask, FREQ_REV_HZ, sigvis)
            out[i]['avgraw'] = avgraw
            out[i]['avg'] = avg
            out[i]['bx'] = compute_calibrated_bx(bxrawAll[i], FREQ_REV_HZ, sigvis)
            recomputed += 1

        print(f"     rows: {len(short_arr)}, matched: {matched}, unmatched: {unmatched}, "
              f"avgraw/avg/bx recomputed from own bxraw+fixed mask: {recomputed}"
              + (" [DRY RUN -- nothing written]" if dry_run else ""))

        if dry_run:
            return

        short_node_name = SHORT_NODE.lstrip('/')
        h5s.remove_node("/", short_node_name)
        new_tab = h5s.create_table("/", short_node_name, out)
        for k, v in donor_attrs.items():
            try:
                setattr(new_tab.attrs, k, v)
            except Exception:
                pass
        print("     rewritten")


def process_fill(fill, runFillsIndex, short_paths=None, dry_run=False,
                  abort_gap_start=ABORT_GAP_START_BX,
                  dirty_bin=DIRTY_BIN_BX,
                  sigvis=DEFAULT_SIGVIS, donor_base=DONOR_BASE):
    """`fill` here only selects WHICH short run files to process (the
    subfolder scdaq2hd5.py originally wrote them into) -- it does NOT limit
    where donor data is searched from; `runFillsIndex` (built ONCE, up
    front, by build_run_fills_index) says which fill directories actually
    matter for each run."""
    short_dir = f"{SHORT_BASE}/{fill}"
    if short_paths is None:
        short_paths = sorted(glob.glob(f"{short_dir}/{fill}_*.hd5"))

    if not short_paths:
        print(f"[SKIP] fill {fill}: no short run file(s) found in {short_dir} "
              f"(expected {fill}_<run>.hd5)")
        return

    print(f"[FILL {fill}]")
    print(f"  short run file(s): {len(short_paths)}")

    for short_path in short_paths:
        process_short_file(short_path, donor_base, runFillsIndex,
                            abort_gap_start=abort_gap_start,
                            dirty_bin=dirty_bin,
                            sigvis=sigvis, dry_run=dry_run)


def fill_from_filename(path):
    """Infer the fill number from a <fill>_<run>.hd5 filename."""
    m = re.match(r'(\d+)_\d+\.hd5$', os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resolve each row's real fillnum from donor (run,ls,nb) data, using a "
                    "one-pass run->fill index (candidates only, no full-tree rescanning per "
                    f"run), merge missing donor columns (node '{DONOR_NODE}') into our "
                    f"converted files (node '{SHORT_NODE}'), and recompute avgraw/avg from our "
                    "own bxraw using a FIXED active-BX mask (BX < --abort-gap-start, excluding "
                    "--dirty-bin). Confirmed fill boundaries are printed as a plain log; no "
                    "automatic pattern/noise-based checks are done here -- verify by hand.")
    parser.add_argument('--file', default=None,
                         help='Process a single specific short .hd5 file instead of scanning '
                              'all of SHORT_BASE. Fill number (for locating the file under '
                              'SHORT_BASE/<fill>/) is inferred from its filename '
                              '(<fill>_<run>.hd5) unless --fill is also given.')
    parser.add_argument('--fill', type=int, default=None,
                         help='Only process run files originally written under '
                              'SHORT_BASE/<fill>/ (unless --file is also given). Donor data is '
                              'still resolved via the global run->fill index for each run.')
    parser.add_argument('--dry-run', action='store_true',
                         help="Print what would happen without writing anything back to disk.")
    parser.add_argument('--abort-gap-start', type=int, default=ABORT_GAP_START_BX,
                         help='BX index at/after which bins are excluded from the fixed '
                              f'active-BX mask used for avgraw/avg (default: {ABORT_GAP_START_BX})')
    parser.add_argument('--dirty-bin', type=int, default=DIRTY_BIN_BX,
                         help='BX index of the one permanently-dirty calibration-pulse-like bin, '
                              f'always excluded from the active-BX mask (default: {DIRTY_BIN_BX})')
    parser.add_argument('--sigvis', type=float, default=DEFAULT_SIGVIS,
                         help=f'Visible cross-section used to calibrate avg = avgraw * '
                              f'{FREQ_REV_HZ} / sigvis (default: {DEFAULT_SIGVIS})')
    parser.add_argument('--donor-base', default=DONOR_BASE,
                         help=f'Base path containing per-fill donor folders, searched globally '
                              f'per run (default: {DONOR_BASE})')
    args = parser.parse_args()

    # Built ONCE, up front: which fill(s) each run appears under in the
    # donor tree. This is the "find all runs present in multiple fills"
    # step, done a single time and reused for every file below instead of
    # re-scanning the whole donor tree per run.
    runFillsIndex = build_run_fills_index(args.donor_base)

    if args.file:
        fill = args.fill if args.fill is not None else fill_from_filename(args.file)
        if fill is None:
            raise SystemExit(f"Could not infer fill number from filename {args.file!r}; "
                              f"pass --fill explicitly.")
        process_fill(fill, runFillsIndex, short_paths=[args.file], dry_run=args.dry_run,
                     abort_gap_start=args.abort_gap_start, dirty_bin=args.dirty_bin,
                     sigvis=args.sigvis, donor_base=args.donor_base)

    elif args.fill is not None:
        process_fill(args.fill, runFillsIndex, dry_run=args.dry_run,
                     abort_gap_start=args.abort_gap_start, dirty_bin=args.dirty_bin,
                     sigvis=args.sigvis, donor_base=args.donor_base)

    else:
        for fill in sorted(x for x in os.listdir(SHORT_BASE) if x.isdigit()):
            process_fill(fill, runFillsIndex, dry_run=args.dry_run,
                         abort_gap_start=args.abort_gap_start, dirty_bin=args.dirty_bin,
                         sigvis=args.sigvis, donor_base=args.donor_base)
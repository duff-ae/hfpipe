#!/usr/bin/env python3
"""
HTCondor submitter for hfcli.run_pipeline.

Creates one submit file containing one job per fill and submits
the whole set with a single condor_submit call.

If RESUBMIT=True, the script scans RESULT_DIR and submits only fills
for which the corresponding output directory does not exist.

Usage:
    module load lxbatch/eossubmit
    python3 condor/run_22.py

The script also loads lxbatch/eossubmit itself for condor_submit,
so loading it manually is optional.
"""

import subprocess
from pathlib import Path


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

CONFIG_YAML = "configs/analysis_22_physics.yaml"

WORKDIR = Path("/eos/user/a/alshevel/hfpipe")
EXECUTABLE = "condor/pipeline.sh"
PROXY_PEM = "condor/proxy.pem"

# "manual"   -> use FILLS below
# "scan_dir" -> use numeric subdirectories under DATA_DIR
FILL_SOURCE = "scan_dir"

FILLS = [8081]

# Raw/input data directories.
DATA_DIR = Path(
    "/eos/cms/store/group/dpg_bril/comm_bril/2022/raw/hfet/"
    #"/eos/cms/store/group/dpg_bril/comm_bril/2022/online/per-bcid/"
)

# ----------------------------------------------------------------------
# Resubmit
# ----------------------------------------------------------------------

# False -> submit all discovered fills
# True  -> submit only fills without an existing result directory
RESUBMIT = False

# Directory containing completed fill directories:
#
# RESULT_DIR/
#   10001/
#   10002/
#   10003/
#
# A fill is considered completed as soon as RESULT_DIR/<fill>/ exists.
RESULT_DIR = Path(
    "/eos/cms/store/group/dpg_bril/comm_bril/2022/reprocessed/hfet_v2/"
)


SUBMIT_DIR = WORKDIR / "condor/submissions"
LOG_DIR = WORKDIR / "condor/logs"

JOB_FLAVOUR = "workday"
ACCOUNTING_GROUP = "group_u_CMS.CAF.COMM"


# ----------------------------------------------------------------------
# Fill discovery
# ----------------------------------------------------------------------

def scan_fill_dirs(data_dir: Path) -> list[int]:
    """Return numeric immediate subdirectories as fill numbers."""
    if not data_dir.is_dir():
        raise RuntimeError(
            f"DATA_DIR does not exist or is not a directory: {data_dir}"
        )

    fills = sorted(
        int(p.name)
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    )

    if not fills:
        raise RuntimeError(
            f"No numeric fill directories found under {data_dir}"
        )

    return fills


def get_fills() -> list[int]:
    if FILL_SOURCE == "manual":
        return sorted(set(FILLS))

    if FILL_SOURCE == "scan_dir":
        return scan_fill_dirs(DATA_DIR)

    raise ValueError(
        f"Unknown FILL_SOURCE: {FILL_SOURCE!r}. "
        "Use 'manual' or 'scan_dir'."
    )


# ----------------------------------------------------------------------
# Resubmit filtering
# ----------------------------------------------------------------------

def filter_missing_fills(fills: list[int]) -> list[int]:
    """
    Keep only fills without an existing output directory.

    RESULT_DIR/<fill>/ existing is treated as evidence that the fill
    completed successfully.
    """
    if not RESULT_DIR.is_dir():
        raise RuntimeError(
            f"RESULT_DIR does not exist or is not a directory: {RESULT_DIR}"
        )

    completed = []
    missing = []

    for fill in fills:
        if (RESULT_DIR / str(fill)).is_dir():
            completed.append(fill)
        else:
            missing.append(fill)

    print(
        f"Resubmit scan:\n"
        f"  discovered:    {len(fills)}\n"
        f"  already done:  {len(completed)}\n"
        f"  to submit:     {len(missing)}"
    )

    if missing:
        print(
            f"Missing fills: "
            f"{missing[:20]}"
            f"{' ...' if len(missing) > 20 else ''}"
        )

    return missing


# ----------------------------------------------------------------------
# Submit files
# ----------------------------------------------------------------------

def write_joblist(path: Path, fills: list[int]) -> None:
    """One fill number per line."""
    with path.open("w") as f:
        for fill in fills:
            f.write(f"{fill}\n")


def write_submit_file(
    submit_file: Path,
    joblist_file: Path,
) -> None:

    content = f"""universe = vanilla

initialdir = {WORKDIR}

executable = {EXECUTABLE}
arguments  = {CONFIG_YAML} $(fill)

output = {LOG_DIR}/fill_$(fill).out
error  = {LOG_DIR}/fill_$(fill).err
log    = {LOG_DIR}/fill_$(fill).log

+JobFlavour = "{JOB_FLAVOUR}"
+AccountingGroup = "{ACCOUNTING_GROUP}"

use_x509userproxy = true
x509userproxy = {PROXY_PEM}

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

queue fill from {joblist_file}
"""

    submit_file.write_text(content)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fills = get_fills()

    # Optional fill range:
    #fills = [f for f in fills if 7920 <= f <= 8113]
    #fills = [f for f in fills if f > 8113]
    fills = [f for f in fills if f > 7920]

    print(
        f"Fill source: {FILL_SOURCE}\n"
        f"Found {len(fills)} fill(s): "
        f"{fills[:10]}{' ...' if len(fills) > 10 else ''}"
    )

    if RESUBMIT:
        fills = filter_missing_fills(fills)

    if not fills:
        print("Nothing to submit: all fill directories already exist.")
        return

    joblist_file = SUBMIT_DIR / "jobs_fills.txt"
    submit_file = SUBMIT_DIR / "fills.sub"

    write_joblist(joblist_file, fills)
    write_submit_file(submit_file, joblist_file)

    print(f"Job list:    {joblist_file}")
    print(f"Submit file: {submit_file}")
    print(f"Submitting {len(fills)} jobs in one transaction...")

    cmd = (
        "module load lxbatch/eossubmit && "
        f"condor_submit {submit_file}"
    )

    res = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        cwd=WORKDIR,
        capture_output=True,
        text=True,
    )

    if res.returncode != 0:
        raise RuntimeError(
            "condor_submit failed\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}"
        )

    print(res.stdout)
    print(f"Submitted {len(fills)} jobs.")


if __name__ == "__main__":
    main()
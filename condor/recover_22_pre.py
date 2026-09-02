#!/usr/bin/env python3
"""
HTCondor submitter for `hfcli.run_pipeline`.
One job per fill (queue fill, job_name from a generated joblist).

module load lxbatch/eossubmit
"""
import subprocess
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration — edit for your setup
# ----------------------------------------------------------------------

CONFIG_YAML = "configs/analysis_22_pre_physics.yaml"
WORKDIR = "/eos/user/a/alshevel/hfpipe"
EXECUTABLE = "condor/pipeline.sh"
PROXY_PEM = "condor/proxy.pem"

# How to get the list of fills — pick ONE:
#   "manual"    -> use the FILLS list below as-is
#   "scan_dir"  -> look at DATA_DIR and take every subfolder whose name is a number
FILL_SOURCE = "scan_dir"

FILLS = [8081]  # used when FILL_SOURCE == "manual"

DATA_DIR = "/eos/cms/store/group/dpg_bril/comm_bril/2022/online/per-bcid/"  # used when FILL_SOURCE == "scan_dir"
# TODO: point DATA_DIR at the folder that contains one subfolder per fill, e.g.:
#   DATA_DIR/10709/...
#   DATA_DIR/10710/...

SUBMIT_DIR = Path("/eos/user/a/alshevel/hfpipe/condor/submissions")
LOG_DIR = Path("/eos/user/a/alshevel/hfpipe/condor/logs")

JOB_FLAVOUR = "workday"  # espresso / microcentury / longlunch / workday / tomorrow ...
ACCOUNTING_GROUP = "group_u_CMS.CAF.COMM"  # TODO: your accounting group

MAX_JOBS_PER_SUBMIT = 1  # split into several .sub files if FILLS is huge

# ----------------------------------------------------------------------


def load_fills_from_file(path: str) -> list[int]:
    """One fill number per line."""
    return [int(ln.strip()) for ln in Path(path).read_text().splitlines() if ln.strip()]


def scan_fill_dirs(data_dir: str) -> list[int]:
    """Take every immediate subfolder of data_dir whose name is a plain number."""
    base = Path(data_dir)
    if not base.is_dir():
        raise RuntimeError(f"DATA_DIR does not exist or is not a directory: {data_dir}")

    fills = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not fills:
        raise RuntimeError(f"No fill subfolders (numeric names) found under {data_dir}")
    return fills


def get_fills() -> list[int]:
    if FILL_SOURCE == "manual":
        return FILLS
    elif FILL_SOURCE == "scan_dir":
        return scan_fill_dirs(DATA_DIR)
    else:
        raise ValueError(f"Unknown FILL_SOURCE: {FILL_SOURCE!r} (use 'manual', 'file' or 'scan_dir')")


def write_submit_file(submit_file: Path, joblist_file: Path) -> None:
    submit_content = f"""universe   = vanilla
executable = {EXECUTABLE}
arguments  = {CONFIG_YAML} $(fill)

output     = {LOG_DIR}/$(job_name).out
error      = {LOG_DIR}/$(job_name).err
log        = {LOG_DIR}/$(job_name).log

+JobFlavour = "{JOB_FLAVOUR}"
+AccountingGroup = "{ACCOUNTING_GROUP}"

use_x509userproxy = true
x509userproxy = {PROXY_PEM}

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

queue fill, job_name from {joblist_file}
"""
    submit_file.write_text(submit_content)


def main() -> None:
    SUBMIT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    fills = sorted(set(get_fills()))
    if not fills:
        raise RuntimeError("No fills to submit")
    
    # Stable beam fills
    fills = [x for x in fills if (x >= 7000 and x <= 8113)]
    #fills = [x for x in fills if (x >= 8115)]

    print(f"Fill source: {FILL_SOURCE} -> {len(fills)} fill(s): {fills[:10]}{' ...' if len(fills) > 10 else ''}")

    n_parts = (len(fills) + MAX_JOBS_PER_SUBMIT - 1) // MAX_JOBS_PER_SUBMIT

    for part in range(n_parts):

        chunk = fills[part * MAX_JOBS_PER_SUBMIT : (part + 1) * MAX_JOBS_PER_SUBMIT]

        joblist_file = SUBMIT_DIR / f"jobs_fills_part{part:03d}.txt"
        with joblist_file.open("w") as jf:
            for fill in chunk:
                job_name = f"fill_{fill}"
                jf.write(f"{fill} {job_name}\n")

        submit_file = SUBMIT_DIR / f"fills_part{part:03d}.sub"
        write_submit_file(submit_file, joblist_file)

        subprocess.run("module load lxbatch/eossubmit", shell=True, executable="/bin/bash")
        res = subprocess.run(["condor_submit", str(submit_file)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"condor_submit failed for part {part:03d}\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        print(f"✅ Submitted part {part + 1}/{n_parts}: {len(chunk)} job(s)")
        print(res.stdout)


if __name__ == "__main__":
    main()
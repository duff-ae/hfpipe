#!/usr/bin/env python3
"""
HTCondor submitter for `hfcli.run_pipeline`.
One job per fill (queue fill, job_name from a generated joblist).

module load lxbatch/eossubmit
"""
import subprocess
import yaml
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration — edit for your setup
# ----------------------------------------------------------------------

CONFIG_YAML = "configs/OC/analysis_oc22_part1.yaml"
WORKDIR = "/cephfs/brilshare/leeja/hfpipe"
EXECUTABLE = "condor/pipeline.sh"

SUBMIT_DIR = Path("/cephfs/brilshare/leeja/hfpipe/condor/submissions")
LOG_DIR = Path("/cephfs/brilshare/leeja/hfpipe/condor/logs")

JOB_FLAVOUR = "workday"  # espresso / microcentury / longlunch / workday / tomorrow ...
ACCOUNTING_GROUP = "group_u_CMS.CAF.COMM"  # TODO: your accounting group

MAX_JOBS_PER_SUBMIT = 1  # split into several .sub files if FILLS is huge

# ----------------------------------------------------------------------

def get_fills() -> list[int]:
    with open(CONFIG_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg['fills']

def write_submit_file(submit_file: Path, joblist_file: Path) -> None:
    submit_content = f"""universe   = vanilla
executable = {EXECUTABLE}
arguments  = {CONFIG_YAML} $(fill)

output     = {LOG_DIR}/$(job_name).out
error      = {LOG_DIR}/$(job_name).err
log        = {LOG_DIR}/$(job_name).log

+JobFlavour = "{JOB_FLAVOUR}"
+AccountingGroup = "{ACCOUNTING_GROUP}"

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
    fills = [8016] #TODO
    
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

        """
        subprocess.run("module load lxbatch/eossubmit", shell=True, executable="/bin/bash")
        res = subprocess.run(["condor_submit", str(submit_file)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"condor_submit failed for part {part:03d}\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        print(f"✅ Submitted part {part + 1}/{n_parts}: {len(chunk)} job(s)")
        print(res.stdout)
        """

if __name__ == "__main__":
    main()

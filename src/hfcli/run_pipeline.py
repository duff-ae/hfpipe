# src/hfcli/run_pipeline.py
from __future__ import annotations

import argparse
import logging

from hfcore.config import load_config
from hfcore.pipeline import run_many_fills


def main():
    parser = argparse.ArgumentParser(description="HF afterglow / Type1 pipeline")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config (e.g. configs/analysis_25_v2.yml)",
    )
    parser.add_argument(
        "--fills",
        nargs="*",
        type=int,
        help="Optional list of fills to process. If not given, uses 'fills' from config.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    run_many_fills(cfg, fills=args.fills)


if __name__ == "__main__":
    main()
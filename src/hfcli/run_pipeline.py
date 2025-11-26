# src/hfcli/run_pipeline.py
from __future__ import annotations

import argparse
import logging
import os

from hfcore.config import load_config
from hfcore.pipeline import run_many_fills
from hfcore.banner import print_md_flag_banner

log = logging.getLogger("hfpipe.run_pipeline")

def parse_fills(arg_list):
    """
    Парсер, который понимает варианты:
      --fills 10463 10470 10471
      --fills 10463-10470
      --fills 10463 10470-10475 10500,10510-10512
    """
    if arg_list is None:
        return None

    fills = []

    for token in arg_list:
        # разрешаем "10463,10470-10475"
        parts = token.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                start = int(start)
                end = int(end)
                fills.extend(range(start, end + 1))
            else:
                fills.append(int(part))

    if not fills:
        return None

    return sorted(set(fills))

def discover_fills(input_dir: str):
    """
    Автоматически находит все филы в input_dir:
      - берём все поддиректории, имя которых состоит только из цифр
      - возвращаем отсортированный список int
    Пример: /cephfs/.../hf_origin/hfet/25/10709/...
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir does not exist or not a dir: {input_dir}")

    fills = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isdir(full) and name.isdigit():
            fills.append(int(name))

    fills = sorted(set(fills))
    if not fills:
        raise RuntimeError(f"discover_fills: no numeric subdirs found in {input_dir}")

    log.info("Auto-discovered fills in %s: %s", input_dir, fills)
    return fills

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
        help="List of fills to process; supports ranges like 10463-11230 and comma-separated lists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    print_md_flag_banner(
        title="HF Analysis & Reprocessing Tool",
        subtitle="BRIL · University of Maryland, College Park",
        version="0.1.0",  # можешь подтянуть из hfcore.__init__.__version__
    )

    cfg = load_config(args.config)

    fills = parse_fills(args.fills)
    
    if fills is None:
        # fills не указан → автоматически берём все филы из input_dir
        fills = discover_fills(cfg.io.input_dir)
        log.info(
            "[run_pipeline] No --fills specified, running over all discovered fills: %s",
            fills,
        )
    else:
        log.info("[run_pipeline] Running over user-specified fills: %s", fills)

    run_many_fills(cfg, fills=fills)

if __name__ == "__main__":
    main()
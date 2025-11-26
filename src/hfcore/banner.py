from __future__ import annotations

import sys
import socket
import platform

RESET = "\033[0m"
BOLD = "\033[1m"

FG_RED    = "\033[91m"
FG_YELLOW = "\033[93m"
FG_WHITE  = "\033[97m"
FG_BLACK  = "\033[30m"

FG_CYAN    = "\033[96m"
FG_MAGENTA = "\033[95m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _px(color: str, ch: str = "█") -> str:
    return f"{color}{ch}{RESET}"


# Оригинальная матрица 16×32
_MD_FLAG_ROWS = [
    "KKKYKYYYKKKYYKKKWWWWWWWWRRRRRRRR",
    "KKKYYKKYKKKYYKKKWWWWWWWRWRRRRRRR",
    "YKKYYKKKYKKYYKKKWWWRWWWRWRRRWRRR",
    "YYYKYKKKYYYKYKKKWRRRRRRRWWWWWWRR",
    "YYYKKYYKYYYKKYYKRWWWWWWWRRRRRRWW",
    "YYYKKYYYKYYKKYYYRRRWRRRWRWWWRWWW",
    "YYYKKYYYKKKYKYYYRRRRRRRWRWWWWWWW",
    "YYYKKYYYKKKYYKKYRRRRRRRRWWWWWWWW",
    "WWWWWWWWRRRRRRRRKKKYKYYYKKKYYKKK",
    "WWWWWWWRWRRRRRRRKKKYYKKYKKKYYKKK",
    "WWWRWWWRWRRRWRRRYKKYYKKKYKKYYKKK",
    "WRRRRRRRWWWWWWWRYYYKYKKKYYYKYKKK",
    "RWWWWWWWRRRRRRRWYYYKKYYKYYYKKYKK",
    "RRRWRRRWRWWWRWWWYYYKKYYYKYYKKYYY",
    "RRRRRRRWRWWWWWWWYYYKKYYYKKKYKYYY",
    "RRRRRRRRWWWWWWWWYYYKKYYYKKKYYKRY",
]


def print_md_flag_banner(
    title: str = "HF Analysis & Reprocessing Tool",
    subtitle: str = "BRIL · University of Maryland, College Park",
    version: str | None = None,
    zoom_x: int = 2,   # <--- увеличение ширины
) -> None:

    if getattr(print_md_flag_banner, "_printed", False):
        return
    setattr(print_md_flag_banner, "_printed", True)

    host = socket.gethostname()
    pyver = platform.python_version()

    if not _supports_color():
        print()
        print(title)
        print(subtitle)
        if version:
            print(f"Version: {version}")
        print(f"Host: {host}")
        print(f"Python: {pyver}")
        print()
        return

    color_map = {
        "K": FG_BLACK,
        "R": FG_RED,
        "Y": FG_YELLOW,
        "W": FG_WHITE,
    }

    indent = "  "
    print()

    # ------ РИСУЕМ ШИРОКИЙ ФЛАГ ------
    for row in _MD_FLAG_ROWS:
        out = []
        for c in row:
            col = color_map.get(c, FG_BLACK)
            # увеличиваем ширину pixel в zoom_x раз
            out.append(_px(col, "█" * zoom_x))
        print(indent + "".join(out))

    # ------ ТЕКСТ ПОД ФЛАГОМ ------
    print()
    print(f"  {BOLD}{FG_RED}{title}{RESET}")
    print(f"  {FG_YELLOW}{subtitle}{RESET}")
    if version:
        print(f"  {FG_CYAN}Version: {version}{RESET}")
    print(f"  {FG_WHITE}Host: {host}{RESET}")
    print(f"  {FG_WHITE}Python: {pyver}{RESET}")
    print(f"  {FG_MAGENTA}HFPIPE · Afterglow + Type-1/Type-2 corrections{RESET}")
    print()

#!/usr/bin/env python3
"""Orchestrator for Figures 3–21 (composite + clean)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporters.paper_figures.config import GenConfig, parse_only_figs
from reporters.paper_figures.data import load_benchmark_bundle
from reporters.paper_figures.style import apply_style
from reporters.paper_figures.figures.figure_03 import generate as gen03
from reporters.paper_figures.figures.figure_04 import generate as gen04
from reporters.paper_figures.figures.figure_05 import generate as gen05
from reporters.paper_figures.figures.figure_06 import generate as gen06
from reporters.paper_figures.figures.figure_07 import generate as gen07
from reporters.paper_figures.figures.figure_08 import generate as gen08
from reporters.paper_figures.figures.figure_09 import generate as gen09
from reporters.paper_figures.figures.figure_10 import generate as gen10
from reporters.paper_figures.figures.figure_11 import generate as gen11
from reporters.paper_figures.figures.figure_12 import generate as gen12
from reporters.paper_figures.figures.figure_13 import generate as gen13
from reporters.paper_figures.figures.figure_14 import generate as gen14
from reporters.paper_figures.figures.figure_15 import generate as gen15
from reporters.paper_figures.figures.figure_16 import generate as gen16
from reporters.paper_figures.figures.figure_17 import generate as gen17
from reporters.paper_figures.figures.figure_18 import generate as gen18
from reporters.paper_figures.figures.figure_19 import generate as gen19
from reporters.paper_figures.figures.figure_20 import generate as gen20
from reporters.paper_figures.figures.figure_21 import generate as gen21

FIGURE_MODULES = [
    (3, gen03),
    (4, gen04),
    (5, gen05),
    (6, gen06),
    (7, gen07),
    (8, gen08),
    (9, gen09),
    (10, gen10),
    (11, gen11),
    (12, gen12),
    (13, gen13),
    (14, gen14),
    (15, gen15),
    (16, gen16),
    (17, gen17),
    (18, gen18),
    (19, gen19),
    (20, gen20),
    (21, gen21),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing cases/results (defaults to cwd discovery).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (composite/ and clean/ will be nested here).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated figure indexes, e.g. 3,5,21",
    )
    parser.add_argument("--no-pdf", action="store_true", help="PNG only for faster previews.")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve() if args.repo_root else ROOT
    only = parse_only_figs(args.only)
    cfg = GenConfig(
        repo_root=repo,
        output_root=args.output,
        figures=sorted(only) if only is not None else None,
        include_pdf=not args.no_pdf,
    )

    bundle = load_benchmark_bundle(cfg.repo_root)
    cfg.ensure_dirs()
    apply_style()

    print(f"[paper_figures] repo root={cfg.repo_root}")
    print(f"[paper_figures] composite→{cfg.output_composite}")
    print(f"[paper_figures] clean     →{cfg.output_clean}")

    for num, generator in FIGURE_MODULES:
        if not cfg.should_generate(num):
            continue
        print(f"Generating Figure {num:02d} …")
        generator(cfg, bundle)

    print("Finished.")


if __name__ == "__main__":
    main()

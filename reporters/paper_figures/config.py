"""Generation configuration shared by figure modules and orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


def _default_repo_root() -> Path:
    # reporters/paper_figures -> reporters -> repo root
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class GenConfig:
    """CLI-driven options for composite + clean figure export."""

    repo_root: Path = field(default_factory=_default_repo_root)
    output_root: Path | None = None  # defaults to reporters/paper_figures/output
    figures: Sequence[int] | None = (
        None  # None => all Figures 3-21 inclusive
    )
    include_pdf: bool = True

    def __post_init__(self) -> None:
        base = (
            Path(self.output_root)
            if self.output_root is not None
            else Path(__file__).resolve().parent / "output"
        )
        self.output_composite = base / "composite"
        self.output_clean = base / "clean"

    def should_generate(self, num: int) -> bool:
        if self.figures is None:
            return True
        return num in frozenset(self.figures)

    def ensure_dirs(self) -> None:
        self.output_composite.mkdir(parents=True, exist_ok=True)
        self.output_clean.mkdir(parents=True, exist_ok=True)


def parse_only_figs(arg: str | None) -> set[int] | None:
    if not arg or not arg.strip():
        return None
    nums: set[int] = set()
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        nums.add(int(part))
    return nums


def results_dir(repo: Path, scenario: int) -> Path:
    return repo / "cases" / "results" / f"scenario-{scenario}"


def latest_json(scenario_dir: Path, glob_pattern: str) -> Path:
    matches = sorted(
        scenario_dir.glob(glob_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No JSON matching {glob_pattern} in {scenario_dir}")
    return matches[0]

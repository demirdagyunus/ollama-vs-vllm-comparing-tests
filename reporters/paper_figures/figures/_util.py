"""Helpers shared by figure modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..config import GenConfig
from ..style import save_figure


SHADE_ALPHA = 0.18


def latency_ms_triplet_s1(row: dict) -> tuple[float, float, float]:
    """Return (p50, p95, p99) latencies in milliseconds for scenario-1 rows."""
    ld = row.get("latency_distribution") or {}
    if ld:
        return (
            float(ld.get("p50", row["avg_latency"])),
            float(ld.get("p95", row["p95_latency"])),
            float(ld.get("p99", row["p99_latency"])),
        )
    return (
        float(row["avg_latency"]),
        float(row["p95_latency"]),
        float(row["p99_latency"]),
    )


def export_composite(fig: plt.Figure, cfg: GenConfig, basename: str) -> None:
    stem = f"{basename}.png"
    pdf_name = f"{basename}.pdf"
    save_figure(
        fig,
        cfg.output_composite / stem,
        cfg.include_pdf,
        cfg.output_composite / pdf_name if cfg.include_pdf else None,
    )
    plt.close(fig)


def export_clean(fig: plt.Figure, cfg: GenConfig, basename: str) -> None:
    stem = f"{basename}.png"
    pdf_name = f"{basename}.pdf"
    save_figure(
        fig,
        cfg.output_clean / stem,
        cfg.include_pdf,
        cfg.output_clean / pdf_name if cfg.include_pdf else None,
    )
    plt.close(fig)


def err_from_min_max(values, mins, maxs) -> np.ndarray:
    """Asymmetric error bars (lower, upper) for matplotlib errorbar yerr."""
    vals = np.asarray(values, dtype=float)
    mins_a = np.asarray(mins, dtype=float)
    maxs_a = np.asarray(maxs, dtype=float)
    lo = np.maximum(vals - mins_a, 0.0)
    hi = np.maximum(maxs_a - vals, 0.0)
    return np.vstack([lo, hi])


def err_upper_tail(values, highs) -> np.ndarray:
    """Upper asymmetric whisker only; lower clamped at 0."""
    vals = np.asarray(values, dtype=float)
    highs_a = np.asarray(highs, dtype=float)
    lo = np.zeros_like(vals)
    hi = np.maximum(highs_a - vals, 0.0)
    return np.vstack([lo, hi])


def percentile_from_row(row: dict[str, Any], key_dist: str, key: str) -> float | None:
    """Return percentile from nested dict or None if absent."""
    d = row.get(key_dist)
    if not isinstance(d, dict) or key not in d:
        return None
    return float(d[key])


def err_from_distribution(
    values: Sequence[float],
    rows: Sequence[dict[str, Any]],
    key_dist: str,
    *,
    low: str = "p25",
    high: str = "p75",
    mid_key: str | None = None,
) -> np.ndarray | None:
    """
    Asymmetric yerr from percentile bounds in row[key_dist] relative to plotted values.

    Error at i: lower = values[i]-low_pct, upper = high_pct-values[i].

    Missing low/high triggers optional fallbacks:

    - If both missing but key_dist dict has ``p50`` and ``p95``, uses conservative
      low = p50 - |p50|*0.5 and high = p95 (latency-style conservative envelope).
    - Else fills from row ``min_latency`` / ``max_latency`` when missing one side.

    ``mid_key`` is reserved for future centering tweaks; asymmetric bars always use plotted
    ``values`` when fallbacks succeed.
    """
    _ = mid_key  # documented API from plan
    vals = np.asarray(values, dtype=float)
    if len(rows) != len(vals):
        return None
    lows_arr: list[float] = []
    highs_arr: list[float] = []
    for row in rows:
        d = row.get(key_dist)
        dl = percentile_from_row(row, key_dist, low) if isinstance(d, dict) else None
        dh = percentile_from_row(row, key_dist, high) if isinstance(d, dict) else None
        if (
            dl is None
            and dh is None
            and isinstance(d, dict)
            and isinstance(d.get("p50"), (int, float))
            and isinstance(d.get("p95"), (int, float))
        ):
            p50_f = float(d["p50"])
            dl = p50_f - 0.5 * abs(p50_f)
            dh = float(d["p95"])
        rl_min = row.get("min_latency")
        rl_max = row.get("max_latency")
        if dl is None and isinstance(rl_min, (int, float)):
            dl = float(rl_min)
        if dh is None and isinstance(rl_max, (int, float)):
            dh = float(rl_max)
        if dl is None or dh is None:
            return None
        lows_arr.append(dl)
        highs_arr.append(dh)

    lows_a = np.asarray(lows_arr, dtype=float)
    highs_a = np.asarray(highs_arr, dtype=float)
    lo_span = vals - lows_a
    hi_span = highs_a - vals
    return np.vstack(
        [
            np.maximum(lo_span, 0.0),
            np.maximum(hi_span, 0.0),
        ]
    )


def shaded_band(
    ax,
    x,
    y_low,
    y_high,
    *,
    color: str,
    alpha: float = SHADE_ALPHA,
    zorder: float = 1,
    **kwargs: Any,
) -> None:
    """Fill_between wrapper for consistent intra-run dispersion bands."""
    ax.fill_between(
        x,
        y_low,
        y_high,
        facecolor=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
        **kwargs,
    )


def rolling_mean_pm_sigma(y: np.ndarray, window: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center-window rolling mean ± 1 sample std along the timeline."""
    yy = np.asarray(y, dtype=float)
    n = len(yy)
    if n == 0:
        return yy, yy, yy
    w = max(3, min(window, n))
    half = w // 2
    mu = np.empty(n)
    lo = np.empty(n)
    hi = np.empty(n)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + (1 if w % 2 else 0))
        seg = yy[s:e]
        if len(seg) < 2:
            m = float(seg.mean()) if seg.size else 0.0
            sig = 0.0
        else:
            m = float(seg.mean())
            sig = float(seg.std(ddof=1))
        mu[i] = m
        lo[i] = m - sig
        hi[i] = m + sig
    return mu, lo, hi


def completion_percentile_band(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """p50–p95 completion time band from persisted distribution (Scenario 2+)."""
    d = row.get("completion_time_distribution") or {}
    if not isinstance(d, dict):
        return None, None
    p50 = d.get("p50")
    p95 = d.get("p95")
    if isinstance(p50, (int, float)) and isinstance(p95, (int, float)):
        return float(p50), float(p95)
    return None, None


def percentile_bar_neighbor_yerr(
    vals: Sequence[float], row: dict[str, Any], pct_keys: Sequence[str]
) -> np.ndarray:
    """Asymmetric bar yerr using adjacent percentiles in completion_time_distribution."""
    dist = row.get("completion_time_distribution") or {}
    errs_lo: list[float] = []
    errs_hi: list[float] = []
    for i, _k in enumerate(pct_keys):
        v = float(vals[i])
        prev_src = pct_keys[i - 1] if i > 0 else None
        next_src = pct_keys[i + 1] if i + 1 < len(pct_keys) else None
        if prev_src is not None and prev_src in dist:
            prev_v = float(dist[prev_src])
        elif isinstance(row.get("min_completion_time"), (int, float)):
            prev_v = float(row["min_completion_time"])
        else:
            prev_v = v
        if next_src is not None and next_src in dist:
            next_v = float(dist[next_src])
        elif isinstance(row.get("max_completion_time"), (int, float)):
            next_v = float(row["max_completion_time"])
        else:
            next_v = v
        errs_lo.append(max(v - prev_v, 0.0))
        errs_hi.append(max(next_v - v, 0.0))
    return np.vstack([np.asarray(errs_lo, dtype=float), np.asarray(errs_hi, dtype=float)])

"""Figure 17: Endurance timeline — CPU & RAM utilisation."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from ..config import GenConfig
from ..style import apply_style
from ._util import (
    SHADE_ALPHA,
    export_clean,
    export_composite,
    rolling_mean_pm_sigma,
    shaded_band,
)


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(17):
        return
    apply_style()
    s4 = bundle["s4"]
    leak = s4.get("memory_leak_analysis") or {}
    end = (s4.get("results") or {}).get("endurance_test_result") or {}
    tl = end.get("resource_usage_timeline") or []
    if not tl:
        return
    ts = np.array([pt["timestamp"] for pt in tl], dtype=float)
    ts_h = (ts - ts.min()) / 3600.0
    mem_pct = np.array([pt["memory"] for pt in tl], dtype=float)
    cpu_pct = np.array([pt["cpu"] for pt in tl], dtype=float)

    lr = linregress(ts_h, mem_pct)

    _, mem_lo, mem_hi = rolling_mean_pm_sigma(mem_pct, 30)
    _, cpu_lo, cpu_hi = rolling_mean_pm_sigma(cpu_pct, 30)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    shaded_band(axes[0], ts_h, mem_lo, mem_hi, color="#4c72b2", alpha=SHADE_ALPHA, zorder=1)
    shaded_band(axes[1], ts_h, cpu_lo, cpu_hi, color="#55a868", alpha=SHADE_ALPHA, zorder=1)
    axes[0].plot(ts_h, mem_pct, label="Measured RAM %", color="#4c72b2", lw=1.8)
    axes[0].plot(
        ts_h,
        lr.intercept + lr.slope * ts_h,
        "--",
        color="orange",
        lw=2,
        label=f"Linear fit ({leak.get('memory_growth_rate_mb_per_hour','?')} MB/h reported)",
    )
    axes[0].set_xlabel("Elapsed time (hours)")
    axes[0].set_ylabel("System memory usage (%)")
    axes[0].set_title("Memory utilisation trajectory")
    axes[0].legend(fontsize=8)

    axes[1].plot(ts_h, cpu_pct, color="#55a868", lw=1.6)
    axes[1].set_xlabel("Elapsed time (hours)")
    axes[1].set_ylabel("CPU utilisation (%)")
    axes[1].set_title("CPU utilisation bursts during endurance cycle")
    fig.suptitle("Resource dynamics during endurance test (Scenario 4)", y=1.03)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure17")

    f1, a1 = plt.subplots(figsize=(6, 3.8))
    _, mlo, mhi = rolling_mean_pm_sigma(mem_pct, 30)
    shaded_band(a1, ts_h, mlo, mhi, color="#4c72b2", alpha=SHADE_ALPHA, zorder=1)
    a1.plot(ts_h, mem_pct, label="Measured", color="#4c72b2")
    a1.plot(ts_h, lr.intercept + lr.slope * ts_h, "--", color="orange", label="Linear fit")
    a1.legend()
    export_clean(f1, cfg, "Figure17_memory_timeline")

    f2, a2 = plt.subplots(figsize=(6, 3.8))
    _, clo, chi = rolling_mean_pm_sigma(cpu_pct, 30)
    shaded_band(a2, ts_h, clo, chi, color="#55a868", alpha=SHADE_ALPHA, zorder=1)
    a2.plot(ts_h, cpu_pct, color="#55a868")
    export_clean(f2, cfg, "Figure17_cpu_timeline")

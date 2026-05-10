"""Figure 13: TTFT percentile distribution & min–max."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style, log_axis_y
from ._util import SHADE_ALPHA, export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(13):
        return
    apply_style()
    s3 = bundle["s3"]
    levels = sorted({r["level"] for r in s3["ollama"]})
    o = {r["level"]: r for r in s3["ollama"]}
    v = {r["level"]: r for r in s3["vllm"]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    pct_keys = ["p50", "p75", "p90", "p95", "p99"]
    labels = pct_keys.copy()
    w = 0.2
    for ax, lvl in zip(axes, levels):
        x = np.arange(len(pct_keys))
        oo, vv = [], []
        for pk in pct_keys:
            oo.append(o[lvl]["ttft_distribution"].get(pk, np.nan))
            vv.append(v[lvl]["ttft_distribution"].get(pk, np.nan))
        ax.bar(x - w / 2, oo, w, label="Ollama", color=OLLAMA_COLOR, edgecolor="white")
        ax.bar(x + w / 2, vv, w, label="vLLM", color=VLLM_COLOR, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        log_axis_y(ax)
        ax.set_title(f"TTFT percentiles ({lvl} concurrent users)")
        ax.set_ylabel("TTFT (seconds, log)")
        ax.legend()

    fig.suptitle("Time-to-first-token distribution (Scenario 3)", y=1.03)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure13_ttft_distribution")

    # Second composite: whisker-style min/max for TTFT range
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lvl in zip(axes2, levels):
        positions = [1, 2]
        mins = [
            float(o[lvl]["min_time_to_first_token"]),
            float(v[lvl]["min_time_to_first_token"]),
        ]
        means = [
            float(o[lvl]["avg_time_to_first_token"]),
            float(v[lvl]["avg_time_to_first_token"]),
        ]
        maxs = [
            float(o[lvl]["max_time_to_first_token"]),
            float(v[lvl]["max_time_to_first_token"]),
        ]
        labs = ["Ollama", "vLLM"]
        for i, lb in enumerate(labs):
            td = o[lvl]["ttft_distribution"] if i == 0 else v[lvl]["ttft_distribution"]
            y_lo_raw = td.get("p25", td.get("p50"))
            y_hi_raw = td.get("p95")
            if isinstance(y_lo_raw, (int, float)) and isinstance(y_hi_raw, (int, float)):
                xl, xr = positions[i] - 0.15, positions[i] + 0.15
                col = OLLAMA_COLOR if i == 0 else VLLM_COLOR
                ax.fill_betweenx(
                    [float(y_lo_raw), float(y_hi_raw)],
                    xl,
                    xr,
                    facecolor=col,
                    alpha=SHADE_ALPHA,
                    zorder=2,
                    linewidth=0,
                )
            ax.plot(
                [positions[i], positions[i]],
                [mins[i], maxs[i]],
                color=OLLAMA_COLOR if i == 0 else VLLM_COLOR,
                lw=6,
                solid_capstyle="round",
                alpha=0.35,
            )
            ax.scatter(positions[i], means[i], s=120, zorder=5, edgecolor="k", linewidth=0.8)
            ax.text(positions[i], maxs[i], f"{maxs[i]:.0f}s", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labs)
        log_axis_y(ax)
        ax.set_title(f"TTFT span ({lvl} users)")
        ax.set_ylabel("TTFT seconds (log scale)")
    fig2.suptitle("TTFT min–mean–max (Scenario 3)", y=1.03)
    fig2.tight_layout()
    export_composite(fig2, cfg, "Figure13_ttft_ranges")

    for lvl in levels:
        fig_c, ax = plt.subplots(figsize=(6, 3.8))
        x = np.arange(len(pct_keys))
        oo = [float(o[lvl]["ttft_distribution"].get(pk)) for pk in pct_keys]
        vv = [float(v[lvl]["ttft_distribution"].get(pk)) for pk in pct_keys]
        ax.bar(x - w / 2, oo, w, label="Ollama", color=OLLAMA_COLOR)
        ax.bar(x + w / 2, vv, w, label="vLLM", color=VLLM_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        log_axis_y(ax)
        ax.set_title(f"TTFT percentiles ({lvl} users)")
        ax.legend()
        export_clean(fig_c, cfg, f"Figure13_percentiles_levels-{lvl}")

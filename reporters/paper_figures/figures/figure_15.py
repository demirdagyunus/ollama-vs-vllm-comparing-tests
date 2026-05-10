"""Figure 15: Stress test — latency vs load with policy threshold."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import VLLM_COLOR, apply_style
from ._util import export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(15):
        return
    apply_style()
    s4 = bundle["s4"]
    cfg_json = s4.get("config") or {}
    threshold = float(cfg_json.get("latency_threshold") or 10.0)
    bp_load = int(s4.get("breaking_point_load") or 50)

    grads = (s4.get("results") or {}).get("gradual_load_results") or []
    if not grads:
        return
    levels = np.array([g["load_level"] for g in grads], dtype=float)
    latency = np.array([g["avg_completion_time"] for g in grads])
    order = np.argsort(levels)
    levels, latency = levels[order], latency[order]

    ymax = max(latency.max(), threshold) * 1.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhspan(0, threshold, color="#d4edda", alpha=0.35, label="Below threshold (target region)")
    ax.axhspan(threshold, ymax, color="#f8d7da", alpha=0.35, label="Beyond threshold (risk region)")
    ax.axhline(threshold, color="crimson", linestyle="--", lw=2, label=f"Policy threshold ({threshold:.0f}s)")
    ax.plot(levels, latency, "o-", color=VLLM_COLOR, lw=2.8, markersize=9, label="Gradual-load average latency")
    ax.scatter(
        [bp_load],
        [latency[-1] if len(latency) else threshold],
        color="darkorange",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
        label=f"Detected breaking load ≈ {bp_load}",
    )

    ax.set_xlabel("Concurrent users (gradual stress phases)")
    ax.set_ylabel("Average completion time (s)")
    ax.set_title("Breaking-point & scalability visualization (Scenario 4 — vLLM)")
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    export_composite(fig, cfg, "Figure15")

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
    ymax = max(latency.max(), threshold) * 1.25
    ax2.axhspan(0, threshold, color="#d4edda", alpha=0.35)
    ax2.axhspan(threshold, ymax, color="#f8d7da", alpha=0.35)
    ax2.axhline(threshold, color="crimson", linestyle="--", lw=2)
    ax2.plot(levels, latency, "o-", color=VLLM_COLOR, lw=2.8, markersize=9)
    ax2.scatter([bp_load], [latency[-1]], color="darkorange", s=200, marker="*", edgecolor="black")
    ax2.set_xlabel("Concurrent users")
    ax2.set_ylabel("Average completion time (s)")
    ax2.set_title("Breaking-point visualization (Scenario 4)")
    ax2.set_ylim(0, ymax)
    fig2.tight_layout()
    export_clean(fig2, cfg, "Figure15_breaking_point")

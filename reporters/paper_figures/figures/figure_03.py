"""Figure 3: Baseline latency distribution (p50, p95, p99) per concurrency."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style, log_axis_y
from ._util import export_clean, export_composite, latency_ms_triplet_s1


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(3):
        return
    apply_style()
    s1 = bundle["s1"]
    o_by = {r["level"]: r for r in s1["ollama"]}
    v_by = {r["level"]: r for r in s1["vllm"]}
    levels = sorted(o_by.keys())

    # --- composite (1x3) ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    pct_labels = ["p50", "p95", "p99"]
    x = np.arange(len(pct_labels))
    w = 0.35
    for ax, lvl in zip(axes, levels):
        oo = latency_ms_triplet_s1(o_by[lvl])
        vv = latency_ms_triplet_s1(v_by[lvl])
        ax.bar(x - w / 2, oo, w, label="Ollama", color=OLLAMA_COLOR, edgecolor="white")
        ax.bar(x + w / 2, vv, w, label="vLLM", color=VLLM_COLOR, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(pct_labels)
        ax.set_title(f"Concurrency level: {lvl}")
        log_axis_y(ax)
        ax.set_ylabel("Latency (ms)")
        ax.legend(loc="upper left", frameon=True)
    axes[0].set_ylim(bottom=1)
    fig.suptitle("Average and percentile latency (Scenario 1: baseline Q&A)", y=1.02)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure03")

    # --- clean panels ---
    for i, lvl in enumerate(levels):
        fig_c, ax = plt.subplots(figsize=(4, 3.5))
        oo = latency_ms_triplet_s1(o_by[lvl])
        vv = latency_ms_triplet_s1(v_by[lvl])
        ax.bar(x - w / 2, oo, w, label="Ollama", color=OLLAMA_COLOR, edgecolor="white")
        ax.bar(x + w / 2, vv, w, label="vLLM", color=VLLM_COLOR, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(pct_labels)
        log_axis_y(ax)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Latency — concurrency {lvl}")
        ax.legend()
        export_clean(fig_c, cfg, f"Figure03_latency_concurrency-{lvl}")

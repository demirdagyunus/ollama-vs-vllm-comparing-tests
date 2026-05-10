"""Figure 8: Reliability — success rate and error rate."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(8):
        return
    apply_style()
    s2 = bundle["s2"]
    levels = sorted({r["level"] for r in s2["ollama"]})
    o = {r["level"]: r for r in s2["ollama"]}
    v = {r["level"]: r for r in s2["vllm"]}

    sr_o = [
        100.0 * o[lv]["successful_tasks"] / max(o[lv]["effective_attempts"], 1)
        for lv in levels
    ]
    sr_v = [
        100.0 * v[lv]["successful_tasks"] / max(v[lv]["total_tasks"], 1)
        for lv in levels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(levels, sr_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    axes[0].plot(levels, sr_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    axes[0].set_xlabel("Concurrent users")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Success rate (effect. attempts for Ollama)")
    axes[0].legend()
    axes[0].set_ylim(-5, 105)

    x = np.arange(len(levels))
    w = 0.35
    eo = [o[lv]["error_rate"] for lv in levels]
    ev = [v[lv]["error_rate"] for lv in levels]
    axes[1].bar(x - w / 2, eo, w, label="Ollama", color=OLLAMA_COLOR, edgecolor="white")
    axes[1].bar(x + w / 2, ev, w, label="vLLM", color=VLLM_COLOR, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(lv) for lv in levels])
    axes[1].set_xlabel("Concurrent users")
    axes[1].set_ylabel("Error rate (%)")
    axes[1].set_title("Error rate by concurrency")
    axes[1].legend()
    fig.suptitle("Reliability under complex reasoning (Scenario 2)", y=1.02)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure08")

    fig_c, axc = plt.subplots(figsize=(4.5, 3.5))
    axc.plot(levels, sr_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    axc.plot(levels, sr_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    axc.set_xlabel("Concurrent users")
    axc.set_ylabel("Success rate (%)")
    axc.legend()
    export_clean(fig_c, cfg, "Figure08_success_rate")

    fig_e, axe = plt.subplots(figsize=(4.5, 3.5))
    axe.bar(x - w / 2, eo, w, label="Ollama", color=OLLAMA_COLOR, edgecolor="white")
    axe.bar(x + w / 2, ev, w, label="vLLM", color=VLLM_COLOR, edgecolor="white")
    axe.set_xticks(x)
    axe.set_xticklabels([str(lv) for lv in levels])
    axe.set_xlabel("Concurrent users")
    axe.set_ylabel("Error rate (%)")
    axe.legend()
    export_clean(fig_e, cfg, "Figure08_error_rate")

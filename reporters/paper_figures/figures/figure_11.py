"""Figure 11: Streaming quality — success vs failures, conversation throughput."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(11):
        return
    apply_style()
    s3 = bundle["s3"]
    levels = sorted({r["level"] for r in s3["ollama"]})
    o = {r["level"]: r for r in s3["ollama"]}
    v = {r["level"]: r for r in s3["vllm"]}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    x = np.arange(len(levels))
    w = 0.35

    succ_o = [
        100 * o[lv]["successful_requests"] / max(o[lv]["total_requests"], 1) for lv in levels
    ]
    succ_v = [
        100 * v[lv]["successful_requests"] / max(v[lv]["total_requests"], 1) for lv in levels
    ]
    err_o = [100 - s for s in succ_o]
    err_v = [100 - s for s in succ_v]

    axes[0].bar(x - w / 2, succ_o, w, label="Success Ollama", color=VLLM_COLOR, alpha=0.95)
    axes[0].bar(x - w / 2, err_o, w, bottom=succ_o, label="Failure Ollama", color=OLLAMA_COLOR)
    axes[0].bar(x + w / 2, succ_v, w, label="Success vLLM", color=VLLM_COLOR, alpha=0.55)
    axes[0].bar(x + w / 2, err_v, w, bottom=succ_v, label="Failure vLLM", color=OLLAMA_COLOR, alpha=0.55)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(lv) for lv in levels])
    axes[0].set_xlabel("Concurrent users")
    axes[0].set_ylabel("Share of requests (%)")
    axes[0].set_title("Success vs failure composition")
    axes[0].legend(fontsize=7)

    # Panel 2: success rate line comparison
    axes[1].plot(levels, succ_o, "o-", color=OLLAMA_COLOR, lw=2, label="Ollama")
    axes[1].plot(levels, succ_v, "s-", color=VLLM_COLOR, lw=2, label="vLLM")
    axes[1].set_xlabel("Concurrent users")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Request-level success rate")
    axes[1].legend()

    conv_o = [o[lv]["throughput_conversations_per_second"] for lv in levels]
    conv_v = [v[lv]["throughput_conversations_per_second"] for lv in levels]
    axes[2].bar(x - w / 2, conv_o, w, label="Ollama", color=OLLAMA_COLOR, edgecolor="white")
    axes[2].bar(x + w / 2, conv_v, w, label="vLLM", color=VLLM_COLOR, edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(lv) for lv in levels])
    axes[2].set_xlabel("Concurrent users")
    axes[2].set_ylabel("Conversations / s")
    axes[2].set_title("Conversation throughput")
    axes[2].legend()

    fig.suptitle("Streaming quality & reliability (Scenario 3)", y=1.03)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure11")

    fc, ax = plt.subplots(figsize=(5, 3.8))
    ax.bar(x - w / 2, succ_o, w, label="Success Ollama", color=VLLM_COLOR)
    ax.bar(x - w / 2, err_o, w, bottom=succ_o, label="Failure Ollama", color=OLLAMA_COLOR, alpha=0.75)
    ax.bar(x + w / 2, succ_v, w, label="Success vLLM", color=VLLM_COLOR, alpha=0.55)
    ax.bar(x + w / 2, err_v, w, bottom=succ_v, label="Failure vLLM", color=OLLAMA_COLOR, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.legend(fontsize=7)
    export_clean(fc, cfg, "Figure11_success_vs_error")

    fl, axl = plt.subplots(figsize=(4.5, 3.5))
    axl.plot(levels, succ_o, "o-", color=OLLAMA_COLOR, lw=2, label="Ollama")
    axl.plot(levels, succ_v, "s-", color=VLLM_COLOR, lw=2, label="vLLM")
    axl.set_xlabel("Concurrent users")
    axl.set_ylabel("Success rate (%)")
    axl.legend()
    export_clean(fl, cfg, "Figure11_success_rate_lines")

    fb, axb = plt.subplots(figsize=(4.5, 3.5))
    axb.bar(x - w / 2, conv_o, w, label="Ollama", color=OLLAMA_COLOR, edgecolor="white")
    axb.bar(x + w / 2, conv_v, w, label="vLLM", color=VLLM_COLOR, edgecolor="white")
    axb.set_xticks(x)
    axb.set_xticklabels([str(lv) for lv in levels])
    axb.legend()
    export_clean(fb, cfg, "Figure11_conversation_throughput")

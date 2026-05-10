"""Figure 14: Queue, bandwidth, memory per request — streaming."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import err_upper_tail, export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(14):
        return
    apply_style()
    s3 = bundle["s3"]
    levels = sorted({r["level"] for r in s3["ollama"]})
    o = {r["level"]: r for r in s3["ollama"]}
    v = {r["level"]: r for r in s3["vllm"]}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    qo = [o[lv]["avg_queue_wait_time"] for lv in levels]
    qv = [v[lv]["avg_queue_wait_time"] for lv in levels]
    bw_o = [o[lv]["avg_bandwidth_per_request"] for lv in levels]
    bw_v = [v[lv]["avg_bandwidth_per_request"] for lv in levels]
    mem_o = [o[lv]["avg_memory_per_request"] for lv in levels]
    mem_v = [v[lv]["avg_memory_per_request"] for lv in levels]

    for ax, (yo, yv, title, ylab) in zip(
        axes,
        [
            (qo, qv, "Avg queue wait time", "Seconds"),
            (bw_o, bw_v, "Bandwidth per request", "KB / request"),
            (mem_o, mem_v, "Memory per request", "MB"),
        ],
    ):
        if "queue wait" in title.lower():
            max_qo = [o[lv]["max_queue_wait_time"] for lv in levels]
            max_qv = [v[lv]["max_queue_wait_time"] for lv in levels]
            ax.errorbar(
                levels,
                yo,
                yerr=err_upper_tail(yo, max_qo),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
            ax.errorbar(
                levels,
                yv,
                yerr=err_upper_tail(yv, max_qv),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        ax.set_title(title)
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel(ylab)
        ax.legend()
    fig.suptitle("System-level streaming indicators (Scenario 3)", y=1.05)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure14")

    for fname, yo, yv, ylab in [
        ("queue_wait", qo, qv, "Seconds"),
        ("bandwidth", bw_o, bw_v, "KB / request"),
        ("memory_per_request", mem_o, mem_v, "MB"),
    ]:
        fig_c, ax = plt.subplots(figsize=(4.5, 3.5))
        if fname == "queue_wait":
            max_qo = [o[lv]["max_queue_wait_time"] for lv in levels]
            max_qv = [v[lv]["max_queue_wait_time"] for lv in levels]
            ax.errorbar(
                levels,
                yo,
                yerr=err_upper_tail(yo, max_qo),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
            ax.errorbar(
                levels,
                yv,
                yerr=err_upper_tail(yv, max_qv),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel(ylab)
        ax.legend()
        export_clean(fig_c, cfg, f"Figure14_{fname}")

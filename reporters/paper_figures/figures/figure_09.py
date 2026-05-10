"""Figure 9: System resources — complex reasoning."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import err_upper_tail, export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(9):
        return
    apply_style()
    s2 = bundle["s2"]
    levels = sorted({r["level"] for r in s2["ollama"]})
    o = {r["level"]: r for r in s2["ollama"]}
    v = {r["level"]: r for r in s2["vllm"]}

    cpu_o = [o[lv]["avg_cpu_usage"] for lv in levels]
    cpu_v = [v[lv]["avg_cpu_usage"] for lv in levels]
    peak_co = [o[lv]["peak_cpu_usage"] for lv in levels]
    peak_cv = [v[lv]["peak_cpu_usage"] for lv in levels]
    mem_rq_o = [o[lv]["avg_memory_per_request"] for lv in levels]
    mem_rq_v = [v[lv]["avg_memory_per_request"] for lv in levels]
    eff_o = [
        o[lv]["avg_tokens_per_second"] / max(o[lv]["avg_cpu_usage"], 1e-6) for lv in levels
    ]
    eff_v = [
        v[lv]["avg_tokens_per_second"] / max(v[lv]["avg_cpu_usage"], 1e-6) for lv in levels
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metric_data = [
        ("Average CPU usage (%)", cpu_o, cpu_v),
        ("Avg memory per request (MB)", mem_rq_o, mem_rq_v),
        ("Efficiency\n(tokens/s per %CPU)", eff_o, eff_v),
    ]
    for ax, (ttl, yo, yv) in zip(axes, metric_data):
        if ttl.startswith("Average CPU"):
            ax.errorbar(
                levels,
                yo,
                yerr=err_upper_tail(cpu_o, peak_co),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
            ax.errorbar(
                levels,
                yv,
                yerr=err_upper_tail(cpu_v, peak_cv),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        ax.set_title(ttl.replace("\n", " "))
        ax.set_xlabel("Concurrent users")
        ax.legend(fontsize=8)
    fig.suptitle("System-level performance — reasoning workloads (Scenario 2)", y=1.06)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure09")

    for (ttl, yo, yv), fname in zip(
        metric_data,
        ["cpu", "memory_per_request", "efficiency_tokens_per_cpu"],
    ):
        fig_c, ax = plt.subplots(figsize=(4.5, 3.5))
        if ttl.startswith("Average CPU"):
            ax.errorbar(
                levels,
                yo,
                yerr=err_upper_tail(cpu_o, peak_co),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
            ax.errorbar(
                levels,
                yv,
                yerr=err_upper_tail(cpu_v, peak_cv),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.55,
                zorder=5,
                capthick=1,
            )
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        ax.set_title(ttl.replace("\n", " "))
        ax.set_xlabel("Concurrent users")
        ax.legend()
        export_clean(fig_c, cfg, f"Figure09_{fname}")

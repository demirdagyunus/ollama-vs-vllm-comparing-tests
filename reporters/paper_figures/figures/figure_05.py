"""Figure 5: CPU, memory and token-efficiency under baseline."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import err_upper_tail, export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(5):
        return
    apply_style()
    s1 = bundle["s1"]
    levels = sorted({r["level"] for r in s1["ollama"]})
    o = {r["level"]: r for r in s1["ollama"]}
    v = {r["level"]: r for r in s1["vllm"]}

    cpu_o = [o[l]["avg_cpu_usage"] for l in levels]
    cpu_v = [v[l]["avg_cpu_usage"] for l in levels]
    mem_o = [o[l]["avg_memory_usage"] for l in levels]
    mem_v = [v[l]["avg_memory_usage"] for l in levels]
    eff_o = [
        o[l]["avg_tokens_per_second"] / max(o[l]["avg_cpu_usage"], 1e-6) for l in levels
    ]
    eff_v = [
        v[l]["avg_tokens_per_second"] / max(v[l]["avg_cpu_usage"], 1e-6) for l in levels
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(levels, cpu_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    axes[0].plot(levels, cpu_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    peak_cpu_o = [o[l]["peak_cpu_usage"] for l in levels]
    peak_cpu_v = [v[l]["peak_cpu_usage"] for l in levels]
    axes[0].errorbar(
        levels,
        cpu_o,
        yerr=err_upper_tail(cpu_o, peak_cpu_o),
        fmt="none",
        ecolor=OLLAMA_COLOR,
        alpha=0.6,
        zorder=5,
        capthick=1,
    )
    axes[0].errorbar(
        levels,
        cpu_v,
        yerr=err_upper_tail(cpu_v, peak_cpu_v),
        fmt="none",
        ecolor=VLLM_COLOR,
        alpha=0.6,
        zorder=5,
        capthick=1,
    )
    axes[0].set_xlabel("Concurrent users")
    axes[0].set_ylabel("CPU utilization (%)")
    axes[0].set_title("Average CPU usage")
    axes[0].legend()

    axes[1].plot(levels, mem_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    axes[1].plot(levels, mem_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    peak_mem_o = [o[l]["peak_memory_usage"] for l in levels]
    peak_mem_v = [v[l]["peak_memory_usage"] for l in levels]
    axes[1].errorbar(
        levels,
        mem_o,
        yerr=err_upper_tail(mem_o, peak_mem_o),
        fmt="none",
        ecolor=OLLAMA_COLOR,
        alpha=0.6,
        zorder=5,
        capthick=1,
    )
    axes[1].errorbar(
        levels,
        mem_v,
        yerr=err_upper_tail(mem_v, peak_mem_v),
        fmt="none",
        ecolor=VLLM_COLOR,
        alpha=0.6,
        zorder=5,
        capthick=1,
    )
    axes[1].set_xlabel("Concurrent users")
    axes[1].set_ylabel("Memory utilization (%)")
    axes[1].set_title("Average memory usage")
    axes[1].legend()

    axes[2].plot(levels, eff_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    axes[2].plot(levels, eff_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    axes[2].set_xlabel("Concurrent users")
    axes[2].set_ylabel("Tokens per second / CPU%")
    axes[2].set_title("Token efficiency (tokens·s⁻¹ / %CPU)")
    axes[2].legend()

    fig.suptitle("Resource utilization and efficiency (Scenario 1)", y=1.02)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure05")

    for name, data_o, data_v, ylab, peaks_o_val, peaks_v_val in [
        ("cpu", cpu_o, cpu_v, "CPU utilization (%)", peak_cpu_o, peak_cpu_v),
        ("memory", mem_o, mem_v, "Memory utilization (%)", peak_mem_o, peak_mem_v),
        ("efficiency", eff_o, eff_v, "Tokens per second / CPU%", None, None),
    ]:
        fig_c, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(levels, data_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, data_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        if peaks_o_val is not None and peaks_v_val is not None:
            ax.errorbar(
                levels,
                data_o,
                yerr=err_upper_tail(data_o, peaks_o_val),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.6,
                zorder=5,
            )
            ax.errorbar(
                levels,
                data_v,
                yerr=err_upper_tail(data_v, peaks_v_val),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.6,
                zorder=5,
            )
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel(ylab)
        ax.legend()
        export_clean(fig_c, cfg, f"Figure05_{name}")

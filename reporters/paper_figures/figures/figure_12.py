"""Figure 12: TTFT-per-CPU and throughput-per-memory proxies."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(12):
        return
    apply_style()
    s3 = bundle["s3"]
    levels = sorted({r["level"] for r in s3["ollama"]})
    o = {r["level"]: r for r in s3["ollama"]}
    v = {r["level"]: r for r in s3["vllm"]}

    ttft_cpu_o = [
        o[lv]["avg_time_to_first_token"] / max(o[lv]["avg_cpu_usage"], 1e-6) for lv in levels
    ]
    ttft_cpu_v = [
        v[lv]["avg_time_to_first_token"] / max(v[lv]["avg_cpu_usage"], 1e-6) for lv in levels
    ]
    tp_mem_o = [
        o[lv]["throughput_requests_per_second"]
        / max(o[lv]["avg_memory_per_request"], 1e-6)
        for lv in levels
    ]
    tp_mem_v = [
        v[lv]["throughput_requests_per_second"]
        / max(v[lv]["avg_memory_per_request"], 1e-6)
        for lv in levels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(levels, ttft_cpu_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    axes[0].plot(levels, ttft_cpu_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    axes[0].set_title("TTFT per CPU utilization (lower is better)")
    axes[0].set_xlabel("Concurrent users")
    axes[0].set_ylabel("Seconds per CPU%")
    axes[0].legend()

    axes[1].plot(levels, tp_mem_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    axes[1].plot(levels, tp_mem_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    axes[1].set_title("Throughput per memory footprint")
    axes[1].set_xlabel("Concurrent users")
    axes[1].set_ylabel("Req/s per MB/request")
    axes[1].legend()
    fig.suptitle("Resource efficiency — streaming workloads (Scenario 3)", y=1.03)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure12")

    f1, a1 = plt.subplots(figsize=(4.8, 3.8))
    a1.plot(levels, ttft_cpu_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    a1.plot(levels, ttft_cpu_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    a1.set_xlabel("Concurrent users")
    a1.set_ylabel("Seconds per CPU%")
    a1.legend()
    export_clean(f1, cfg, "Figure12_ttft_per_cpu")

    f2, a2 = plt.subplots(figsize=(4.8, 3.8))
    a2.plot(levels, tp_mem_o, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
    a2.plot(levels, tp_mem_v, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
    a2.set_xlabel("Concurrent users")
    a2.set_ylabel("Req/s per MB/request")
    a2.legend()
    export_clean(f2, cfg, "Figure12_throughput_per_memory")

"""Figure 4: Baseline throughput, latency, token rate vs concurrency."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..data import latency_distribution_q
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import err_from_min_max, export_clean, export_composite


def _rows_by_level(entries: list[dict]) -> dict[int, dict]:
    return {r["level"]: r for r in entries}


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(4):
        return
    apply_style()
    s1 = bundle["s1"]
    levels = sorted(_rows_by_level(s1["ollama"]).keys())
    o = _rows_by_level(s1["ollama"])
    v = _rows_by_level(s1["vllm"])

    def series(key_ollama_vllama: tuple[str, str]):
        ko, kv = key_ollama_vllama
        return [o[l][ko] for l in levels], [v[l][kv] for l in levels]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Throughput req/s
    yo, yv = series(("throughput", "throughput"))
    axes[0].plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", linewidth=2)
    axes[0].plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", linewidth=2)
    axes[0].set_xlabel("Concurrent users")
    axes[0].set_ylabel("Throughput (req/s)")
    axes[0].set_title("Request throughput")
    axes[0].legend()
    # Avg latency ms with intra-run dispersion (min–avg–p95)
    yo = [o[l]["avg_latency"] for l in levels]
    yv = [v[l]["avg_latency"] for l in levels]
    min_o = [o[l]["min_latency"] for l in levels]
    min_v = [v[l]["min_latency"] for l in levels]
    p95_o = [
        float(latency_distribution_q(o[l], "p95") or o[l]["p95_latency"]) for l in levels
    ]
    p95_v = [
        float(latency_distribution_q(v[l], "p95") or v[l]["p95_latency"]) for l in levels
    ]
    err_o = err_from_min_max(yo, min_o, p95_o)
    err_v = err_from_min_max(yv, min_v, p95_v)
    axes[1].errorbar(
        levels,
        yo,
        yerr=err_o,
        fmt="none",
        ecolor=OLLAMA_COLOR,
        alpha=0.55,
        zorder=4,
        capthick=1,
    )
    axes[1].errorbar(
        levels,
        yv,
        yerr=err_v,
        fmt="none",
        ecolor=VLLM_COLOR,
        alpha=0.55,
        zorder=4,
        capthick=1,
    )
    axes[1].plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", linewidth=2)
    axes[1].plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", linewidth=2)
    axes[1].set_xlabel("Concurrent users")
    axes[1].set_ylabel("Avg latency (ms)")
    axes[1].set_title("Average latency")
    axes[1].legend()
    # Tokens/s
    yo = [o[l]["avg_tokens_per_second"] for l in levels]
    yv = [v[l]["avg_tokens_per_second"] for l in levels]
    axes[2].plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", linewidth=2)
    axes[2].plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", linewidth=2)
    axes[2].set_xlabel("Concurrent users")
    axes[2].set_ylabel("Tokens / s")
    axes[2].set_title("Token generation rate")
    axes[2].legend()
    fig.suptitle("Comparative baseline performance (Scenario 1)", y=1.02)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure04")

    titles = [
        ("throughput", "Throughput (req/s)", ("throughput", "throughput")),
        ("avg_latency", "Average latency (ms)", None),
        ("tokens", "Tokens / s", ("avg_tokens_per_second", "avg_tokens_per_second")),
    ]
    for name, ylab, keys in titles:
        fig_c, ax = plt.subplots(figsize=(4.5, 3.5))
        if keys:
            yo, yv = series(keys)  # type: ignore[arg-type]
        else:
            yo = [o[l]["avg_latency"] for l in levels]
            yv = [v[l]["avg_latency"] for l in levels]
            min_oc = [o[l]["min_latency"] for l in levels]
            min_vc = [v[l]["min_latency"] for l in levels]
            p95_oc = [
                float(latency_distribution_q(o[l], "p95") or o[l]["p95_latency"]) for l in levels
            ]
            p95_vc = [
                float(latency_distribution_q(v[l], "p95") or v[l]["p95_latency"]) for l in levels
            ]
            ax.errorbar(
                levels,
                yo,
                yerr=err_from_min_max(yo, min_oc, p95_oc),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.55,
                zorder=4,
            )
            ax.errorbar(
                levels,
                yv,
                yerr=err_from_min_max(yv, min_vc, p95_vc),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.55,
                zorder=4,
            )
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", linewidth=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", linewidth=2)
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel(ylab)
        ax.legend()
        export_clean(fig_c, cfg, f"Figure04_{name}")

"""Figure 20: Scenario-level snapshot across six KPI families."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(20):
        return
    apply_style()
    sm = bundle["summary"]
    labels = ["S1 Baseline", "S2 Reasoning", "S3 Streaming", "S4 Stress"]

    sr_o = [
        sm["scenario1"]["ollama"]["success_rate"],
        sm["scenario2"]["ollama"]["success_rate"],
        sm["scenario3"]["ollama"]["success_rate"],
        0.0,
    ]
    sr_v = [
        sm["scenario1"]["vllm"]["success_rate"],
        sm["scenario2"]["vllm"]["success_rate"],
        sm["scenario3"]["vllm"]["success_rate"],
        sm["scenario4"]["vllm"].get("success_rate", 0.0),
    ]
    tp_o = [
        sm["scenario1"]["ollama"]["throughput_rps"],
        sm["scenario2"]["ollama"]["throughput_rps"],
        sm["scenario3"]["ollama"]["throughput_rps"],
        0.0,
    ]
    tp_v = [
        sm["scenario1"]["vllm"]["throughput_rps"],
        sm["scenario2"]["vllm"]["throughput_rps"],
        sm["scenario3"]["vllm"]["throughput_rps"],
        sm["scenario4"]["vllm"].get("throughput_rps", 0.0),
    ]
    lat_o = [
        sm["scenario1"]["ollama"]["avg_latency_ms"],
        sm["scenario2"]["ollama"]["avg_latency_ms"],
        sm["scenario3"]["ollama"]["avg_latency_ms"],
        0.0,
    ]
    lat_v = [
        sm["scenario1"]["vllm"]["avg_latency_ms"],
        sm["scenario2"]["vllm"]["avg_latency_ms"],
        sm["scenario3"]["vllm"]["avg_latency_ms"],
        sm["scenario4"]["vllm"].get("avg_latency_ms", 0.0),
    ]
    tk_o = [
        sm["scenario1"]["ollama"]["tokens_per_sec"],
        sm["scenario2"]["ollama"]["tokens_per_sec"],
        sm["scenario3"]["ollama"]["tokens_per_sec"],
        0.0,
    ]
    tk_v = [
        sm["scenario1"]["vllm"]["tokens_per_sec"],
        sm["scenario2"]["vllm"]["tokens_per_sec"],
        sm["scenario3"]["vllm"]["tokens_per_sec"],
        sm["scenario4"]["vllm"].get("tokens_per_sec", 0.0),
    ]
    cpu_o = [
        sm["scenario1"]["ollama"]["cpu"],
        sm["scenario2"]["ollama"]["cpu"],
        sm["scenario3"]["ollama"]["cpu"],
        sm["scenario4"]["vllm"].get("cpu", 0.0),
    ]
    cpu_v = [
        sm["scenario1"]["vllm"]["cpu"],
        sm["scenario2"]["vllm"]["cpu"],
        sm["scenario3"]["vllm"]["cpu"],
        sm["scenario4"]["vllm"].get("cpu", 0.0),
    ]
    er_o = [
        sm["scenario1"]["ollama"]["error_rate"],
        sm["scenario2"]["ollama"]["error_rate"],
        sm["scenario3"]["ollama"]["error_rate"],
        0.0,
    ]
    er_v = [
        sm["scenario1"]["vllm"]["error_rate"],
        sm["scenario2"]["vllm"]["error_rate"],
        sm["scenario3"]["vllm"]["error_rate"],
        sm["scenario4"]["vllm"].get("error_rate", 0.0),
    ]

    sr_o_std = [
        sm["scenario1"]["ollama"].get("success_rate_std", 0.0),
        sm["scenario2"]["ollama"].get("success_rate_std", 0.0),
        sm["scenario3"]["ollama"].get("success_rate_std", 0.0),
        0.0,
    ]
    sr_v_std = [
        sm["scenario1"]["vllm"].get("success_rate_std", 0.0),
        sm["scenario2"]["vllm"].get("success_rate_std", 0.0),
        sm["scenario3"]["vllm"].get("success_rate_std", 0.0),
        sm["scenario4"]["vllm"].get("success_rate_std", 0.0),
    ]
    tp_o_std = [
        sm["scenario1"]["ollama"].get("throughput_rps_std", 0.0),
        sm["scenario2"]["ollama"].get("throughput_rps_std", 0.0),
        sm["scenario3"]["ollama"].get("throughput_rps_std", 0.0),
        0.0,
    ]
    tp_v_std = [
        sm["scenario1"]["vllm"].get("throughput_rps_std", 0.0),
        sm["scenario2"]["vllm"].get("throughput_rps_std", 0.0),
        sm["scenario3"]["vllm"].get("throughput_rps_std", 0.0),
        sm["scenario4"]["vllm"].get("throughput_rps_std", 0.0),
    ]
    lat_o_std = [
        sm["scenario1"]["ollama"].get("avg_latency_ms_std", 0.0),
        sm["scenario2"]["ollama"].get("avg_latency_ms_std", 0.0),
        sm["scenario3"]["ollama"].get("avg_latency_ms_std", 0.0),
        0.0,
    ]
    lat_v_std = [
        sm["scenario1"]["vllm"].get("avg_latency_ms_std", 0.0),
        sm["scenario2"]["vllm"].get("avg_latency_ms_std", 0.0),
        sm["scenario3"]["vllm"].get("avg_latency_ms_std", 0.0),
        sm["scenario4"]["vllm"].get("avg_latency_ms_std", 0.0),
    ]
    tk_o_std = [
        sm["scenario1"]["ollama"].get("tokens_per_sec_std", 0.0),
        sm["scenario2"]["ollama"].get("tokens_per_sec_std", 0.0),
        sm["scenario3"]["ollama"].get("tokens_per_sec_std", 0.0),
        0.0,
    ]
    tk_v_std = [
        sm["scenario1"]["vllm"].get("tokens_per_sec_std", 0.0),
        sm["scenario2"]["vllm"].get("tokens_per_sec_std", 0.0),
        sm["scenario3"]["vllm"].get("tokens_per_sec_std", 0.0),
        sm["scenario4"]["vllm"].get("tokens_per_sec_std", 0.0),
    ]
    cpu_o_std = [
        sm["scenario1"]["ollama"].get("cpu_std", 0.0),
        sm["scenario2"]["ollama"].get("cpu_std", 0.0),
        sm["scenario3"]["ollama"].get("cpu_std", 0.0),
        sm["scenario4"]["vllm"].get("cpu_std", 0.0),
    ]
    cpu_v_std = [
        sm["scenario1"]["vllm"].get("cpu_std", 0.0),
        sm["scenario2"]["vllm"].get("cpu_std", 0.0),
        sm["scenario3"]["vllm"].get("cpu_std", 0.0),
        sm["scenario4"]["vllm"].get("cpu_std", 0.0),
    ]
    er_o_std = [
        sm["scenario1"]["ollama"].get("error_rate_std", 0.0),
        sm["scenario2"]["ollama"].get("error_rate_std", 0.0),
        sm["scenario3"]["ollama"].get("error_rate_std", 0.0),
        0.0,
    ]
    er_v_std = [
        sm["scenario1"]["vllm"].get("error_rate_std", 0.0),
        sm["scenario2"]["vllm"].get("error_rate_std", 0.0),
        sm["scenario3"]["vllm"].get("error_rate_std", 0.0),
        sm["scenario4"]["vllm"].get("error_rate_std", 0.0),
    ]

    def panel(
        ax,
        o_vals,
        v_vals,
        title: str,
        ylabel: str,
        log_scale: bool = False,
        *,
        std_o=None,
        std_v=None,
    ):
        x = np.arange(len(labels))
        w = 0.38
        bar_kw_o: dict = {"edgecolor": "white"}
        bar_kw_v: dict = {"edgecolor": "white"}
        if std_o is not None:
            bar_kw_o["yerr"] = np.asarray(std_o, dtype=float)
            bar_kw_o["capsize"] = 3
        if std_v is not None:
            bar_kw_v["yerr"] = np.asarray(std_v, dtype=float)
            bar_kw_v["capsize"] = 3
        ax.bar(x - w / 2, o_vals, w, label="Ollama", color=OLLAMA_COLOR, **bar_kw_o)
        ax.bar(x + w / 2, v_vals, w, label="vLLM", color=VLLM_COLOR, **bar_kw_v)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        if log_scale:
            ax.set_yscale("symlog", linthresh=1)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9))
    panel(
        axes[0, 0],
        sr_o,
        sr_v,
        "Success rate (effective comp.)",
        "%",
        std_o=sr_o_std,
        std_v=sr_v_std,
    )
    panel(
        axes[0, 1],
        tp_o,
        tp_v,
        "Throughput (native units)",
        "req·task / s",
        std_o=tp_o_std,
        std_v=tp_v_std,
    )
    panel(
        axes[0, 2],
        lat_o,
        lat_v,
        "Average latency emphasis",
        "ms",
        log_scale=True,
        std_o=lat_o_std,
        std_v=lat_v_std,
    )
    panel(
        axes[1, 0],
        tk_o,
        tk_v,
        "Token cadence",
        "tokens / s",
        std_o=tk_o_std,
        std_v=tk_v_std,
    )
    panel(
        axes[1, 1],
        cpu_o,
        cpu_v,
        "CPU utilisation snapshots",
        "%",
        std_o=cpu_o_std,
        std_v=cpu_v_std,
    )
    panel(
        axes[1, 2],
        er_o,
        er_v,
        "Measured error prevalence",
        "%",
        log_scale=True,
        std_o=er_o_std,
        std_v=er_v_std,
    )

    axes[1, 2].annotate(
        "S4 Stress only documents vLLM",
        xy=(3, er_v[-1]),
        fontsize=8,
        ha="center",
    )

    fig.suptitle("Scenario-wise KPI overview (figures align with Tables 7 in manuscript)", y=0.995)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure20")

    f1, a1 = plt.subplots(figsize=(7, 3.8))
    panel(
        a1,
        sr_o,
        sr_v,
        "Success rates",
        "%",
        std_o=sr_o_std,
        std_v=sr_v_std,
    )
    export_clean(f1, cfg, "Figure20_success_rates")

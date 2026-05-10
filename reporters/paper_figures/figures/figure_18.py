"""Figure 18: Stress overview — aggregated panels."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import VLLM_COLOR, apply_style
from ._util import err_upper_tail, export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(18):
        return
    apply_style()
    s4 = bundle["s4"]
    res = s4.get("results") or {}
    grad_list = res.get("gradual_load_results") or []
    end = res.get("endurance_test_result") or {}
    if not grad_list or not end:
        return
    grad = grad_list[0]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    phases = ["Gradual", "Endurance"]
    loads = [grad["load_level"], end["load_level"]]
    lat_vals = [grad["avg_completion_time"], end["avg_completion_time"]]
    lat_peaks = [grad["p95_completion_time"], end["p95_completion_time"]]
    axes[0, 0].bar(
        phases,
        lat_vals,
        color=[VLLM_COLOR, "#2c7873"],
        edgecolor="white",
        yerr=err_upper_tail(lat_vals, lat_peaks),
        capsize=3,
    )
    axes[0, 0].set_title("Avg completion latency")
    axes[0, 0].set_ylabel("Seconds")

    axes[0, 1].bar(phases, [100 - grad["error_rate"], 100 - end["error_rate"]], color="#5DA5DA")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_title("Success rate proxy")
    axes[0, 1].set_ylabel("%")

    res_cmp = ["CPU %", "Mem %", "GPU %"]
    grad_res = [
        grad["avg_cpu_usage"],
        grad["avg_memory_usage"],
        grad.get("avg_gpu_usage") or 0,
    ]
    end_res = [
        end["avg_cpu_usage"],
        end["avg_memory_usage"],
        end.get("avg_gpu_usage") or 0,
    ]
    x = np.arange(len(res_cmp))
    w = 0.35
    peak_grad = [
        grad["peak_cpu_usage"],
        grad["peak_memory_usage"],
        grad.get("peak_gpu_usage") or 0,
    ]
    peak_end = [
        end["peak_cpu_usage"],
        end["peak_memory_usage"],
        end.get("peak_gpu_usage") or 0,
    ]
    axes[0, 2].bar(
        x - w / 2,
        grad_res,
        w,
        label="Gradual",
        color=VLLM_COLOR,
        yerr=err_upper_tail(grad_res, peak_grad),
        capsize=3,
    )
    axes[0, 2].bar(
        x + w / 2,
        end_res,
        w,
        label="Endurance",
        color="#2c7873",
        alpha=0.85,
        yerr=err_upper_tail(end_res, peak_end),
        capsize=3,
    )
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(res_cmp)
    axes[0, 2].set_title("Avg resource footprints")
    axes[0, 2].legend()

    axes[1, 0].bar(phases, [loads[0], loads[1]], color="#E24A33", edgecolor="white")
    axes[1, 0].set_title("Concurrent users")
    axes[1, 0].set_ylabel("Users")

    axes[1, 1].bar(
        phases,
        [grad["avg_response_quality"], end["avg_response_quality"]],
        color="#FAB15C",
        edgecolor="white",
        label="Quality",
    )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Tracked response quality proxy")
    xs = np.array([0, 1])
    axes[1, 1].plot(
        xs,
        [grad["consistency_score"], end["consistency_score"]],
        "purple",
        marker="x",
        markersize=10,
        linewidth=1,
        linestyle="",
        label="Consistency",
    )
    axes[1, 1].legend(fontsize=7)

    degrade = [
        grad["performance_degradation_percent"],
        end["performance_degradation_percent"],
    ]
    axes[1, 2].bar(phases, degrade, color="#A6CEE3", edgecolor="white")
    axes[1, 2].set_title("Performance delta vs baseline")
    axes[1, 2].set_ylabel("% change")

    fig.suptitle("Comprehensive stress-characterisation mosaic (Scenario 4 — vLLM)", y=1.02)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure18")

    fl, axl = plt.subplots(figsize=(4.8, 3.8))
    axl.bar(
        phases,
        lat_vals,
        color=[VLLM_COLOR, "#2c7873"],
        yerr=err_upper_tail(lat_vals, lat_peaks),
        capsize=3,
    )
    axl.set_ylabel("Seconds")
    export_clean(fl, cfg, "Figure18_latency_comparison")

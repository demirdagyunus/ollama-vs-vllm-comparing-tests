"""Figure 16: Gradual vs endurance — latency & token throughput distributions."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import VLLM_COLOR, apply_style
from ._util import export_clean, export_composite, percentile_bar_neighbor_yerr


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(16):
        return
    apply_style()
    s4 = bundle["s4"]
    res = s4.get("results") or {}
    grad_list = res.get("gradual_load_results") or []
    end = res.get("endurance_test_result")
    if not grad_list or not end:
        return
    grad = grad_list[0]

    g_dist = grad.get("completion_time_distribution") or {}
    e_dist = end.get("completion_time_distribution") or {}
    pct_keys = ["p50", "p75", "p90", "p95", "p99"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(pct_keys))
    w = 0.38
    g_vals = [float(g_dist.get(k, np.nan)) for k in pct_keys]
    e_vals = [float(e_dist.get(k, np.nan)) for k in pct_keys]
    yerr_grad = percentile_bar_neighbor_yerr(g_vals, grad, pct_keys)
    yerr_end = percentile_bar_neighbor_yerr(e_vals, end, pct_keys)
    axes[0].bar(
        x - w / 2,
        g_vals,
        w,
        label="Gradual surge",
        color=VLLM_COLOR,
        alpha=0.85,
        yerr=yerr_grad,
        capsize=2,
    )
    axes[0].bar(
        x + w / 2,
        e_vals,
        w,
        label="Endurance baseline",
        color="#2c7873",
        alpha=0.85,
        yerr=yerr_end,
        capsize=2,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pct_keys)
    axes[0].set_ylabel("Completion time (s)")
    axes[0].set_title("Latency percentiles — gradual vs endurance")
    axes[0].legend()

    # Token throughput proxies: avg min max from gradual/endurance payload
    cat_labels = ["Gradual", "Endurance"]
    tps_mid = [
        grad["avg_tokens_per_second"],
        end["avg_tokens_per_second"],
    ]
    errs_low = np.array(tps_mid) - np.array(
        [grad["min_tokens_per_second"], end["min_tokens_per_second"]]
    )
    errs_high = (
        np.array([grad["max_tokens_per_second"], end["max_tokens_per_second"]]) - np.array(tps_mid)
    )
    axes[1].bar(
        cat_labels,
        tps_mid,
        yerr=[errs_low, errs_high],
        capsize=5,
        color=[VLLM_COLOR, "#2c7873"],
        edgecolor="white",
    )
    axes[1].set_ylabel("Tokens/s (native min-to-max dispersion)")
    axes[1].set_title("Token throughput dispersion")
    fig.suptitle("Gradual-load vs endurance stress behaviour (Scenario 4)", y=1.03)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure16")

    f1, a1 = plt.subplots(figsize=(6, 3.8))
    a1.bar(
        x - w / 2,
        g_vals,
        w,
        label="Gradual",
        color=VLLM_COLOR,
        yerr=percentile_bar_neighbor_yerr(g_vals, grad, pct_keys),
        capsize=2,
    )
    a1.bar(
        x + w / 2,
        e_vals,
        w,
        label="Endurance",
        color="#2c7873",
        alpha=0.85,
        yerr=percentile_bar_neighbor_yerr(e_vals, end, pct_keys),
        capsize=2,
    )
    a1.set_xticks(x)
    a1.set_xticklabels(pct_keys)
    a1.legend()
    export_clean(f1, cfg, "Figure16_latency_percentiles")

    f2, a2 = plt.subplots(figsize=(5, 3.8))
    a2.bar(cat_labels, tps_mid, yerr=[errs_low, errs_high], capsize=5, color=[VLLM_COLOR, "#2c7873"])
    export_clean(f2, cfg, "Figure16_token_throughput")

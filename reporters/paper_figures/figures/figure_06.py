"""Figure 6: Complex reasoning task-level comparison."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import (
    completion_percentile_band,
    err_upper_tail,
    export_clean,
    export_composite,
    shaded_band,
)


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(6):
        return
    apply_style()
    s2 = bundle["s2"]
    levels = sorted({r["level"] for r in s2["ollama"]})

    def by_lvl(side: str) -> dict[int, dict]:
        return {r["level"]: r for r in s2[side]}

    o, v = by_lvl("ollama"), by_lvl("vllm")

    sr_o = [
        100.0 * o[lv]["successful_tasks"] / max(o[lv]["effective_attempts"], 1)
        for lv in levels
    ]
    sr_v = [
        100.0 * v[lv]["successful_tasks"] / max(v[lv]["total_tasks"], 1)
        for lv in levels
    ]

    metrics: list[tuple[str, list[float], list[float]]] = [
        ("Success rate (%)\n(effect. attempts)", sr_o, sr_v),
        ("Throughput\n(tasks/s)", [o[lv]["throughput"] for lv in levels], [v[lv]["throughput"] for lv in levels]),
        ("Avg completion time (s)", [o[lv]["avg_completion_time"] for lv in levels], [v[lv]["avg_completion_time"] for lv in levels]),
        ("Avg queue wait (s)", [o[lv]["avg_queue_wait_time"] for lv in levels], [v[lv]["avg_queue_wait_time"] for lv in levels]),
        ("Tokens / s", [o[lv]["avg_tokens_per_second"] for lv in levels], [v[lv]["avg_tokens_per_second"] for lv in levels]),
        ("Error rate (%)", [o[lv]["error_rate"] for lv in levels], [v[lv]["error_rate"] for lv in levels]),
    ]

    def decorate_axes(ax, metric_title: str, yo: list[float], yv: list[float]) -> None:
        ttl = metric_title.lower()
        if "completion time" in ttl:
            pb_o = [completion_percentile_band(o[lv]) for lv in levels]
            pb_v = [completion_percentile_band(v[lv]) for lv in levels]
            if all(a[0] is not None and a[1] is not None for a in pb_o + pb_v):
                po = [float(a[0]) for a in pb_o]
                p9o = [float(a[1]) for a in pb_o]
                pv = [float(a[0]) for a in pb_v]
                p9v = [float(a[1]) for a in pb_v]
                shaded_band(ax, levels, po, p9o, color=OLLAMA_COLOR, zorder=1)
                shaded_band(ax, levels, pv, p9v, color=VLLM_COLOR, zorder=1)
        if "queue wait" in ttl:
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

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (title, yo, yv) in zip(axes.flatten(), metrics):
        decorate_axes(ax, title, yo, yv)
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2, zorder=3)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2, zorder=3)
        ax.set_title(title.replace("\n", " "))
        ax.set_xlabel("Concurrent users")
        ax.legend(fontsize=8)
    fig.suptitle("Complex reasoning workloads (Scenario 2)", y=1.01)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure06")

    clean_names = [
        "success_rate",
        "throughput",
        "completion_time",
        "queue_wait",
        "tokens_per_sec",
        "error_rate",
    ]
    for (title, yo, yv), cname in zip(metrics, clean_names):
        fig_c, ax = plt.subplots(figsize=(4.5, 3.5))
        decorate_axes(ax, title, yo, yv)
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        ax.set_xlabel("Concurrent users")
        ax.set_title(title.replace("\n", " "))
        ax.legend()
        export_clean(fig_c, cfg, f"Figure06_{cname}")

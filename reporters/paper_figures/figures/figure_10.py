"""Figure 10: Streaming performance overview."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import err_from_min_max, export_clean, export_composite


def _tps(r: dict) -> float:
    t = r.get("avg_tokens_per_second") or 0.0
    if t > 0:
        return float(t)
    dur = max(r.get("test_duration") or 1.0, 1e-6)
    return float(r.get("total_tokens_generated") or 0) / dur


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(10):
        return
    apply_style()
    s3 = bundle["s3"]
    levels = sorted({r["level"] for r in s3["ollama"]})
    o = {r["level"]: r for r in s3["ollama"]}
    v = {r["level"]: r for r in s3["vllm"]}

    def pct_succ(side: dict, lv: int) -> float:
        r = side[lv]
        return 100.0 * r["successful_requests"] / max(r["total_requests"], 1)

    def pct_conv(side: dict, lv: int) -> float:
        r = side[lv]
        return 100.0 * r["completed_conversations"] / max(r["total_conversations"], 1)

    panels = [
        ("Request success rate (%)", lambda d, lv: pct_succ(d, lv)),
        ("Avg TTFT (s)", lambda d, lv: d[lv]["avg_time_to_first_token"]),
        ("Throughput (req/s)", lambda d, lv: d[lv]["throughput_requests_per_second"]),
        ("Conversation completion (%)", lambda d, lv: pct_conv(d, lv)),
        ("Avg completion time (s)", lambda d, lv: d[lv]["avg_completion_time"]),
        ("Tokens / s", lambda d, lv: _tps(d[lv])),
    ]

    def plot_streaming(ax, title: str, fn) -> None:
        yo = [fn(o, lv) for lv in levels]
        yv = [fn(v, lv) for lv in levels]
        if title.startswith("Avg TTFT"):
            mo = [float(o[lv]["min_time_to_first_token"]) for lv in levels]
            xo = [float(o[lv]["max_time_to_first_token"]) for lv in levels]
            mv = [float(v[lv]["min_time_to_first_token"]) for lv in levels]
            xv = [float(v[lv]["max_time_to_first_token"]) for lv in levels]
            ax.errorbar(
                levels,
                yo,
                yerr=err_from_min_max(yo, mo, xo),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.55,
                zorder=4,
                capthick=1,
            )
            ax.errorbar(
                levels,
                yv,
                yerr=err_from_min_max(yv, mv, xv),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.55,
                zorder=4,
                capthick=1,
            )
        elif title.startswith("Avg completion time"):
            mo = [float(o[lv]["min_completion_time"]) for lv in levels]
            xo = [float(o[lv]["max_completion_time"]) for lv in levels]
            mv = [float(v[lv]["min_completion_time"]) for lv in levels]
            xv = [float(v[lv]["max_completion_time"]) for lv in levels]
            ax.errorbar(
                levels,
                yo,
                yerr=err_from_min_max(yo, mo, xo),
                fmt="none",
                ecolor=OLLAMA_COLOR,
                alpha=0.55,
                zorder=4,
                capthick=1,
            )
            ax.errorbar(
                levels,
                yv,
                yerr=err_from_min_max(yv, mv, xv),
                fmt="none",
                ecolor=VLLM_COLOR,
                alpha=0.55,
                zorder=4,
                capthick=1,
            )
        elif title.startswith("Tokens / s"):
            mo = [float(o[lv].get("min_tokens_per_second") or 0) for lv in levels]
            xo = [float(o[lv].get("max_tokens_per_second") or 0) for lv in levels]
            mv = [float(v[lv].get("min_tokens_per_second") or 0) for lv in levels]
            xv = [float(v[lv].get("max_tokens_per_second") or 0) for lv in levels]
            if any(abs(x) > 1e-9 for x in mo + xo + mv + xv):
                ax.errorbar(
                    levels,
                    yo,
                    yerr=err_from_min_max(yo, mo, xo),
                    fmt="none",
                    ecolor=OLLAMA_COLOR,
                    alpha=0.55,
                    zorder=4,
                    capthick=1,
                )
                ax.errorbar(
                    levels,
                    yv,
                    yerr=err_from_min_max(yv, mv, xv),
                    fmt="none",
                    ecolor=VLLM_COLOR,
                    alpha=0.55,
                    zorder=4,
                    capthick=1,
                )
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2, zorder=5)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2, zorder=5)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (title, fn) in zip(axes.flatten(), panels):
        plot_streaming(ax, title, fn)
        ax.set_title(title)
        ax.set_xlabel("Concurrent users")
        ax.legend(fontsize=8)
    fig.suptitle("Streaming conversational workloads (Scenario 3)", y=1.02)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure10")

    clean_names = [
        "success_rate",
        "ttft",
        "throughput_rps",
        "conversation_completion",
        "completion_time",
        "tokens_per_sec",
    ]
    for idx, cname in enumerate(clean_names):
        fig_c, ax = plt.subplots(figsize=(4.5, 3.5))
        title_clean, fn = panels[idx]
        plot_streaming(ax, title_clean, fn)
        ax.set_xlabel("Concurrent users")
        ax.legend()
        export_clean(fig_c, cfg, f"Figure10_{cname}")

"""Figure 19: Specialty four-panel analytic summary."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import err_from_min_max, export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(19):
        return
    apply_style()
    s1, s2, s3 = bundle["s1"], bundle["s2"], bundle["s3"]
    o2 = [r for r in s2["ollama"]]
    v2 = [r for r in s2["vllm"]]

    def mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    def std_acc_pct(rows: list[dict], key: str) -> float:
        vs = [100.0 * float(r[key]) for r in rows]
        return float(np.std(np.array(vs, dtype=float), ddof=1)) if len(vs) > 1 else 0.0

    math_o = mean([x["math_accuracy"] * 100 for x in o2])
    code_o = mean([x["code_accuracy"] * 100 for x in o2])
    qa_o = mean([x["qa_accuracy"] * 100 for x in o2])
    math_v = mean([x["math_accuracy"] * 100 for x in v2])
    code_v = mean([x["code_accuracy"] * 100 for x in v2])
    qa_v = mean([x["qa_accuracy"] * 100 for x in v2])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    cats = ["Math", "Code", "QA"]
    x = np.arange(3)
    w = 0.35
    axes[0, 0].bar(
        x - w / 2,
        [math_o, code_o, qa_o],
        w,
        label="Ollama",
        color=OLLAMA_COLOR,
        yerr=[
            std_acc_pct(o2, "math_accuracy"),
            std_acc_pct(o2, "code_accuracy"),
            std_acc_pct(o2, "qa_accuracy"),
        ],
        capsize=3,
    )
    axes[0, 0].bar(
        x + w / 2,
        [math_v, code_v, qa_v],
        w,
        label="vLLM",
        color=VLLM_COLOR,
        yerr=[
            std_acc_pct(v2, "math_accuracy"),
            std_acc_pct(v2, "code_accuracy"),
            std_acc_pct(v2, "qa_accuracy"),
        ],
        capsize=3,
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(cats)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("Mean reasoning accuracy (Scenario 2)")
    axes[0, 0].legend()

    o3 = s3["ollama"]
    v3 = s3["vllm"]
    lv = sorted({r["level"] for r in o3})
    ttft_o = [
        float(next(r["avg_time_to_first_token"] * 1000 for r in o3 if r["level"] == l)) for l in lv
    ]
    ttft_v = [
        float(next(r["avg_time_to_first_token"] * 1000 for r in v3 if r["level"] == l)) for l in lv
    ]
    min_ttft_o = [
        float(next(r["min_time_to_first_token"] for r in o3 if r["level"] == l)) * 1000 for l in lv
    ]
    max_ttft_o = [
        float(next(r["max_time_to_first_token"] for r in o3 if r["level"] == l)) * 1000 for l in lv
    ]
    min_ttft_v = [
        float(next(r["min_time_to_first_token"] for r in v3 if r["level"] == l)) * 1000 for l in lv
    ]
    max_ttft_v = [
        float(next(r["max_time_to_first_token"] for r in v3 if r["level"] == l)) * 1000 for l in lv
    ]
    axes[0, 1].errorbar(
        lv,
        ttft_o,
        yerr=err_from_min_max(ttft_o, min_ttft_o, max_ttft_o),
        fmt="none",
        ecolor=OLLAMA_COLOR,
        alpha=0.55,
        zorder=4,
        capthick=1,
    )
    axes[0, 1].errorbar(
        lv,
        ttft_v,
        yerr=err_from_min_max(ttft_v, min_ttft_v, max_ttft_v),
        fmt="none",
        ecolor=VLLM_COLOR,
        alpha=0.55,
        zorder=4,
        capthick=1,
    )
    axes[0, 1].plot(lv, ttft_o, "o-", color=OLLAMA_COLOR, lw=2, label="Ollama")
    axes[0, 1].plot(lv, ttft_v, "s-", color=VLLM_COLOR, lw=2, label="vLLM")
    axes[0, 1].set_xlabel("Concurrent users")
    axes[0, 1].set_ylabel("Avg TTFT (ms)")
    axes[0, 1].set_title("Streaming TTFT curves (Scenario 3)")
    axes[0, 1].legend()

    # Throughput ladders across scenarios
    scen_labels = ["Baseline\n(S1)", "Reasoning\n(S2)", "Streaming\n(S3)"]
    tp_o = [
        mean([r["throughput"] for r in s1["ollama"]]),
        mean([r["throughput"] for r in s2["ollama"]]),
        mean([r["throughput_requests_per_second"] for r in s3["ollama"]]),
    ]
    tp_v = [
        mean([r["throughput"] for r in s1["vllm"]]),
        mean([r["throughput"] for r in s2["vllm"]]),
        mean([r["throughput_requests_per_second"] for r in s3["vllm"]]),
    ]
    xm = np.arange(len(scen_labels))
    axes[1, 0].bar(xm - w / 2, tp_o, w, label="Ollama", color=OLLAMA_COLOR)
    axes[1, 0].bar(xm + w / 2, tp_v, w, label="vLLM", color=VLLM_COLOR)
    axes[1, 0].set_xticks(xm)
    axes[1, 0].set_xticklabels(scen_labels)
    axes[1, 0].set_ylabel("Throughput (native units)")
    axes[1, 0].set_title("Cross-scenario throughput snapshot")
    axes[1, 0].legend()

    eff_o = [
        mean(
            [
                r["avg_tokens_per_second"] / max(r["avg_cpu_usage"], 1e-6)
                for r in s1["ollama"]
            ]
        ),
        mean(
            [
                r["avg_tokens_per_second"] / max(r["avg_cpu_usage"], 1e-6)
                for r in s2["ollama"]
            ]
        ),
        mean(
            [
                r["avg_tokens_per_second"] / max(r["avg_cpu_usage"], 1e-6)
                for r in s3["ollama"]
            ]
        ),
    ]
    eff_v = [
        mean(
            [
                r["avg_tokens_per_second"] / max(r["avg_cpu_usage"], 1e-6)
                for r in s1["vllm"]
            ]
        ),
        mean(
            [
                r["avg_tokens_per_second"] / max(r["avg_cpu_usage"], 1e-6)
                for r in s2["vllm"]
            ]
        ),
        mean(
            [
                r["avg_tokens_per_second"] / max(r["avg_cpu_usage"], 1e-6)
                for r in s3["vllm"]
            ]
        ),
    ]
    axes[1, 1].bar(xm - w / 2, eff_o, w, label="Ollama", color=OLLAMA_COLOR)
    axes[1, 1].bar(xm + w / 2, eff_v, w, label="vLLM", color=VLLM_COLOR)
    axes[1, 1].set_xticks(xm)
    axes[1, 1].set_xticklabels(scen_labels)
    axes[1, 1].set_ylabel("Tokens per second per %CPU")
    axes[1, 1].set_title("Computational efficiency")
    axes[1, 1].legend()

    fig.suptitle("Specialised cross-scenario juxtaposition", y=0.995)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure19")

    f1, a1 = plt.subplots(figsize=(5, 3.8))
    a1.bar(
        x - w / 2,
        [math_o, code_o, qa_o],
        w,
        label="Ollama",
        color=OLLAMA_COLOR,
        yerr=[
            std_acc_pct(o2, "math_accuracy"),
            std_acc_pct(o2, "code_accuracy"),
            std_acc_pct(o2, "qa_accuracy"),
        ],
        capsize=3,
    )
    a1.bar(
        x + w / 2,
        [math_v, code_v, qa_v],
        w,
        label="vLLM",
        color=VLLM_COLOR,
        yerr=[
            std_acc_pct(v2, "math_accuracy"),
            std_acc_pct(v2, "code_accuracy"),
            std_acc_pct(v2, "qa_accuracy"),
        ],
        capsize=3,
    )
    a1.set_xticks(x)
    a1.set_xticklabels(["Math", "Code", "QA"])
    a1.legend()
    export_clean(f1, cfg, "Figure19_reasoning_accuracy")

    ft, at = plt.subplots(figsize=(4.8, 3.8))
    at.errorbar(
        lv,
        ttft_o,
        yerr=err_from_min_max(ttft_o, min_ttft_o, max_ttft_o),
        fmt="none",
        ecolor=OLLAMA_COLOR,
        alpha=0.55,
        zorder=4,
        capthick=1,
    )
    at.errorbar(
        lv,
        ttft_v,
        yerr=err_from_min_max(ttft_v, min_ttft_v, max_ttft_v),
        fmt="none",
        ecolor=VLLM_COLOR,
        alpha=0.55,
        zorder=4,
        capthick=1,
    )
    at.plot(lv, ttft_o, "o-", color=OLLAMA_COLOR, lw=2, label="Ollama")
    at.plot(lv, ttft_v, "s-", color=VLLM_COLOR, lw=2, label="vLLM")
    at.set_xlabel("Concurrent users")
    at.set_ylabel("TTFT (ms)")
    at.legend()
    export_clean(ft, cfg, "Figure19_ttft")

    ftb, atb = plt.subplots(figsize=(5, 3.8))
    atb.bar(xm - w / 2, tp_o, w, label="Ollama", color=OLLAMA_COLOR)
    atb.bar(xm + w / 2, tp_v, w, label="vLLM", color=VLLM_COLOR)
    atb.set_xticks(xm)
    atb.set_xticklabels(["S1", "S2", "S3"])
    atb.legend()
    export_clean(ftb, cfg, "Figure19_throughput_ladder")

    fe, ae = plt.subplots(figsize=(5, 3.8))
    ae.bar(xm - w / 2, eff_o, w, label="Ollama", color=OLLAMA_COLOR)
    ae.bar(xm + w / 2, eff_v, w, label="vLLM", color=VLLM_COLOR)
    ae.set_xticks(xm)
    ae.set_xticklabels(["S1", "S2", "S3"])
    ae.legend()
    export_clean(fe, cfg, "Figure19_efficiency")

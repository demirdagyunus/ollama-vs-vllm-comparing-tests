"""Figure 7: Task-type reasoning accuracy."""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import export_clean, export_composite


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(7):
        return
    apply_style()
    s2 = bundle["s2"]
    levels = sorted({r["level"] for r in s2["ollama"]})
    o = {r["level"]: r for r in s2["ollama"]}
    v = {r["level"]: r for r in s2["vllm"]}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    keys = [
        ("math_accuracy", "Math accuracy"),
        ("code_accuracy", "Code accuracy"),
        ("qa_accuracy", "QA accuracy"),
    ]
    for ax, (k, ttl) in zip(axes, keys):
        yo = [o[lv][k] * 100 for lv in levels]
        yv = [v[lv][k] * 100 for lv in levels]
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        ax.set_title(ttl)
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.set_ylim(0, 100)
    fig.suptitle("Task-type accuracy (Scenario 2)", y=1.05)
    fig.tight_layout()
    export_composite(fig, cfg, "Figure07")

    for (k, ttl) in keys:
        fig_c, ax = plt.subplots(figsize=(4.5, 3.5))
        yo = [o[lv][k] * 100 for lv in levels]
        yv = [v[lv][k] * 100 for lv in levels]
        ax.plot(levels, yo, "o-", color=OLLAMA_COLOR, label="Ollama", lw=2)
        ax.plot(levels, yv, "s-", color=VLLM_COLOR, label="vLLM", lw=2)
        ax.set_title(ttl)
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        export_clean(fig_c, cfg, f"Figure07_{k.replace('_accuracy','')}")

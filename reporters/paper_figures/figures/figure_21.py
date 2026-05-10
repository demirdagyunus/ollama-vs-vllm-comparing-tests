"""Figure 21: Normalised radar + advantage ratios (baseline / reasoning / streaming)."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ..config import GenConfig
from ..style import OLLAMA_COLOR, VLLM_COLOR, apply_style
from ._util import export_clean, export_composite


def _norm_max(a: float, b: float) -> tuple[float, float]:
    m = max(a, b, 1e-12)
    return a / m, b / m


def generate(cfg: GenConfig, bundle: dict) -> None:
    if not cfg.should_generate(21):
        return
    apply_style()
    sm = bundle["summary"]

    # Radar uses six composite indicators (scenario 1+2 aggregates)
    s1o, s1v = sm["scenario1"]["ollama"], sm["scenario1"]["vllm"]
    s2o, s2v = sm["scenario2"]["ollama"], sm["scenario2"]["vllm"]

    throughput_o = np.mean([s1o["throughput_rps"], s2o["throughput_rps"]])
    throughput_v = np.mean([s1v["throughput_rps"], s2v["throughput_rps"]])
    success_o = np.mean([s1o["success_rate"], s2o["success_rate"]])
    success_v = np.mean([s1v["success_rate"], s2v["success_rate"]])
    latency_o = s1o["avg_latency_ms"] + s2o["avg_latency_ms"]
    latency_v = s1v["avg_latency_ms"] + s2v["avg_latency_ms"]
    inv_lat_o = 1.0 / max(latency_o, 1e-12)
    inv_lat_v = 1.0 / max(latency_v, 1e-12)
    tokens_o = np.mean([s1o["tokens_per_sec"], s2o["tokens_per_sec"]])
    tokens_v = np.mean([s1v["tokens_per_sec"], s2v["tokens_per_sec"]])
    cpu_eff_o = np.mean([
        s1o["tokens_per_sec"] / max(s1o["cpu"], 1e-6),
        s2o["tokens_per_sec"] / max(s2o["cpu"], 1e-6),
    ])
    cpu_eff_v = np.mean([
        s1v["tokens_per_sec"] / max(s1v["cpu"], 1e-6),
        s2v["tokens_per_sec"] / max(s2v["cpu"], 1e-6),
    ])
    mem_proxy_o = tokens_o / max(s2o["error_rate"], 0.001)
    mem_proxy_v = tokens_v / max(s2v["error_rate"], 0.001)

    dims = []
    dims.append(("Throughput", *_norm_max(throughput_o, throughput_v)))
    dims.append(("Success", *_norm_max(success_o, success_v)))
    dims.append(("Low_latency", *_norm_max(inv_lat_o, inv_lat_v)))
    dims.append(("Token_eff.", *_norm_max(tokens_o, tokens_v)))
    dims.append(("CPU_eff.", *_norm_max(cpu_eff_o, cpu_eff_v)))
    dims.append(("Mem_eff.", *_norm_max(mem_proxy_o, mem_proxy_v)))

    o_vec = np.array([d[1] for d in dims] + [dims[0][1]])
    v_vec = np.array([d[2] for d in dims] + [dims[0][2]])
    labels = [d[0] for d in dims]
    theta = np.linspace(0, 2 * np.pi, len(o_vec), endpoint=True)

    fig = plt.figure(figsize=(13, 5.5))
    ax0 = plt.subplot(1, 2, 1, polar=True)
    ax0.plot(theta, o_vec, "o-", color=OLLAMA_COLOR, linewidth=2, label="Ollama composite")
    ax0.fill(theta, o_vec, color=OLLAMA_COLOR, alpha=0.18)
    ax0.plot(theta, v_vec, "s-", color=VLLM_COLOR, linewidth=2, label="vLLM composite")
    ax0.fill(theta, v_vec, color=VLLM_COLOR, alpha=0.18)
    ax0.set_xticks(theta[:-1])
    ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylim(0, 1.05)
    ax0.legend(loc="upper left", bbox_to_anchor=(1.05, 1.08))
    ax0.set_title("Normalised service radar", pad=24)

    ax1 = plt.subplot(1, 2, 2)
    scenarios_x = ["S1", "S2", "S3"]

    def ratio_tp(i: str) -> float:
        o = float(sm[f"scenario{i}"]["ollama"]["throughput_rps"])
        v = float(sm[f"scenario{i}"]["vllm"]["throughput_rps"])
        return max(v / max(o, 1e-9), 1e-3)

    def ratio_sr(i: str) -> float:
        o = float(sm[f"scenario{i}"]["ollama"]["success_rate"])
        v = float(sm[f"scenario{i}"]["vllm"]["success_rate"])
        return max(v / max(o, 1e-6), 1e-3)

    def ratio_lat(i: str) -> float:
        o = float(sm[f"scenario{i}"]["ollama"]["avg_latency_ms"])
        v = float(sm[f"scenario{i}"]["vllm"]["avg_latency_ms"])
        return max(o / max(v, 1e-9), 1e-3)

    idx = np.arange(3)
    width = 0.25
    ax1.bar(
        idx - width,
        [ratio_tp("1"), ratio_tp("2"), ratio_tp("3")],
        width,
        label="Throughput fav vLLM (×)",
        color="#c7e9c0",
    )
    ax1.bar(
        idx,
        [ratio_sr("1"), ratio_sr("2"), ratio_sr("3")],
        width,
        label="Success-rate fav vLLM (×)",
        color="#aec7e8",
    )
    ax1.bar(
        idx + width,
        [ratio_lat("1"), ratio_lat("2"), ratio_lat("3")],
        width,
        label="Latency fav lower vLLM (Oll/VLL)",
        color="#fdd0a2",
    )
    ax1.set_xticks(idx)
    ax1.set_xticklabels(scenarios_x)
    ax1.set_yscale("log")
    ax1.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("Approximate multiplier (>=1 favours vLLM)")
    ax1.set_title("Scenario-wise dominance ratios")
    ax1.legend(fontsize=8)

    fig.suptitle("Radar + ratio juxtaposition mirroring manuscript Figure 21", y=1.02)
    fig.tight_layout(w_pad=2.5)
    export_composite(fig, cfg, "Figure21")

    fig2 = plt.figure(figsize=(8, 4))
    axp = plt.subplot(projection="polar")
    axp.plot(theta, o_vec, "o-", color=OLLAMA_COLOR, linewidth=2)
    axp.plot(theta, v_vec, "s-", color=VLLM_COLOR, linewidth=2)
    axp.legend(["Ollama", "vLLM"], fontsize=10, loc="lower left", bbox_to_anchor=(0.92, -0.05))
    export_clean(fig2, cfg, "Figure21_radar_only")

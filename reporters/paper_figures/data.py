"""Load benchmark JSON artefacts and derive cross-scenario summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import latest_json, results_dir


def latency_distribution_q(row: dict[str, Any], pct_key: str) -> float | None:
    """Read latency_distribution['pXX'] if present."""
    ld = row.get("latency_distribution") or {}
    if pct_key not in ld:
        return None
    return float(ld[pct_key])


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_scenario1(repo: Path) -> dict[str, list[dict[str, Any]]]:
    sdir = results_dir(repo, 1)
    return {
        "ollama": _load_json(latest_json(sdir, "ollama_performance_data_*.json"))[
            "results"
        ],
        "vllm": _load_json(latest_json(sdir, "vllm_performance_data_*.json"))[
            "results"
        ],
    }


def service_unreachable_errors(r: dict[str, Any]) -> int:
    total = 0
    et = r.get("error_types") or {}
    for msg, cnt in et.items():
        if "connect" in msg.lower() or "servname" in msg.lower() or "errno" in msg.lower():
            total += cnt
    return total


def load_scenario2(repo: Path) -> dict[str, Any]:
    sdir = results_dir(repo, 2)
    ollama_doc = _load_json(latest_json(sdir, "ollama_complex_reasoning_data_*.json"))
    vllm_doc = _load_json(latest_json(sdir, "vllm_complex_reasoning_data_*.json"))

    def enrich_rows(doc_list: list[dict]) -> list[dict]:
        rows = []
        for r in doc_list:
            su = sum(
                c
                for k, c in (r.get("error_types") or {}).items()
                if "failed to connect" in k.lower()
                or "servname" in k.lower()
                or "not known" in k.lower()
            )
            if su == 0:
                su = service_unreachable_errors(r)
            tot = int(r.get("total_tasks", 0))
            eff = max(0, tot - su)
            rows.append({**r, "service_unreachable": su, "effective_attempts": eff})
        return rows

    o_res = enrich_rows(ollama_doc["results"])
    v_res = enrich_rows(vllm_doc["results"])
    return {"ollama": o_res, "vllm": v_res, "ollama_meta": ollama_doc, "vllm_meta": vllm_doc}


def load_scenario3(repo: Path) -> dict[str, list[dict[str, Any]]]:
    sdir = results_dir(repo, 3)
    return {
        "ollama": _load_json(latest_json(sdir, "ollama_streaming_data_*.json"))[
            "results"
        ],
        "vllm": _load_json(latest_json(sdir, "vllm_streaming_data_*.json"))["results"],
    }


def load_scenario4(repo: Path) -> dict[str, Any]:
    sdir = results_dir(repo, 4)
    doc = _load_json(latest_json(sdir, "vllm_stress_test_data_*.json"))
    return doc


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.std(np.array(xs, dtype=float), ddof=1))


def load_overall_summary(repo: Path) -> dict[str, Any]:
    s1 = load_scenario1(repo)
    s2 = load_scenario2(repo)
    s3 = load_scenario3(repo)
    s4 = load_scenario4(repo)
    return _build_summary(s1, s2, s3, s4)


def load_benchmark_bundle(repo: Path) -> dict[str, Any]:
    """Single-pass load for figure generation."""
    s1 = load_scenario1(repo)
    s2 = load_scenario2(repo)
    s3 = load_scenario3(repo)
    s4 = load_scenario4(repo)
    return {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "summary": _build_summary(s1, s2, s3, s4),
    }


def _build_summary(s1: dict, s2: dict, s3: dict, s4_doc: dict) -> dict[str, Any]:
    def s1_metrics(rows: list[dict]) -> dict[str, Any]:
        sr_list = [
            100.0 * r["successful_requests"] / max(r["total_requests"], 1) for r in rows
        ]
        lat_list = [r["avg_latency"] for r in rows]
        tp_list = [r["throughput"] for r in rows]
        tps_list = [r["avg_tokens_per_second"] for r in rows]
        er_list = [r["error_rate"] for r in rows]
        cpu_list = [r["avg_cpu_usage"] for r in rows]
        return {
            "success_rate": mean(sr_list),
            "success_rate_std": _std(sr_list),
            "avg_latency_ms": mean(lat_list),
            "avg_latency_ms_std": _std(lat_list),
            "throughput_rps": mean(tp_list),
            "throughput_rps_std": _std(tp_list),
            "tokens_per_sec": mean(tps_list),
            "tokens_per_sec_std": _std(tps_list),
            "error_rate": mean(er_list),
            "error_rate_std": _std(er_list),
            "cpu": mean(cpu_list),
            "cpu_std": _std(cpu_list),
        }

    def s2_ollama_row() -> dict[str, Any]:
        o = s2["ollama"]
        tot_tasks = sum(int(r["total_tasks"]) for r in o)
        tot_unreachable = sum(int(r["service_unreachable"]) for r in o)
        eff = max(1, tot_tasks - tot_unreachable)
        succ = sum(int(r["successful_tasks"]) for r in o)
        sr_levels = [
            100.0 * r["successful_tasks"] / max(r["effective_attempts"], 1) for r in o
        ]
        lat_ms = [r["avg_completion_time"] * 1000 for r in o]
        tp = [r["throughput"] for r in o]
        tps = [r["avg_tokens_per_second"] for r in o]
        er = [r["error_rate"] for r in o]
        cpu = [r["avg_cpu_usage"] for r in o]
        return {
            "success_rate": 100.0 * succ / eff,
            "success_rate_std": _std(sr_levels),
            "avg_latency_ms": mean(lat_ms),
            "avg_latency_ms_std": _std(lat_ms),
            "throughput_rps": mean(tp),
            "throughput_rps_std": _std(tp),
            "tokens_per_sec": mean(tps),
            "tokens_per_sec_std": _std(tps),
            "error_rate": mean(er),
            "error_rate_std": _std(er),
            "cpu": mean(cpu),
            "cpu_std": _std(cpu),
            "total_tasks": float(tot_tasks),
            "service_unreachable": float(tot_unreachable),
        }

    def s2_vllm_row() -> dict[str, Any]:
        v = s2["vllm"]
        sr_levels = [
            100.0 * r["successful_tasks"] / max(r["total_tasks"], 1) for r in v
        ]
        lat_ms = [r["avg_completion_time"] * 1000 for r in v]
        tp = [r["throughput"] for r in v]
        tps = [r["avg_tokens_per_second"] for r in v]
        er = [r["error_rate"] for r in v]
        cpu = [r["avg_cpu_usage"] for r in v]
        return {
            "success_rate": 100.0
            * sum(r["successful_tasks"] for r in v)
            / max(sum(r["total_tasks"] for r in v), 1),
            "success_rate_std": _std(sr_levels),
            "avg_latency_ms": mean(lat_ms),
            "avg_latency_ms_std": _std(lat_ms),
            "throughput_rps": mean(tp),
            "throughput_rps_std": _std(tp),
            "tokens_per_sec": mean(tps),
            "tokens_per_sec_std": _std(tps),
            "error_rate": mean(er),
            "error_rate_std": _std(er),
            "cpu": mean(cpu),
            "cpu_std": _std(cpu),
        }

    def s3_metrics(rows: list[dict]) -> dict[str, Any]:
        sr_list = [
            100.0 * r["successful_requests"] / max(r["total_requests"], 1) for r in rows
        ]
        lat_ms = [r["avg_completion_time"] * 1000 for r in rows]
        tp_list = [r["throughput_requests_per_second"] for r in rows]
        tps_list = [r["avg_tokens_per_second"] for r in rows]
        er_list = [r["error_rate"] for r in rows]
        cpu_list = [r["avg_cpu_usage"] for r in rows]
        ttft_list = [r["avg_time_to_first_token"] * 1000 for r in rows]
        return {
            "success_rate": mean(sr_list),
            "success_rate_std": _std(sr_list),
            "avg_latency_ms": mean(lat_ms),
            "avg_latency_ms_std": _std(lat_ms),
            "throughput_rps": mean(tp_list),
            "throughput_rps_std": _std(tp_list),
            "tokens_per_sec": mean(tps_list),
            "tokens_per_sec_std": _std(tps_list),
            "error_rate": mean(er_list),
            "error_rate_std": _std(er_list),
            "cpu": mean(cpu_list),
            "cpu_std": _std(cpu_list),
            "ttft_ms": mean(ttft_list),
            "ttft_ms_std": _std(ttft_list),
        }

    grad = (s4_doc.get("results") or {}).get("gradual_load_results") or []
    endurance = (s4_doc.get("results") or {}).get("endurance_test_result") or {}
    stress_vllm: dict[str, Any] = {}

    g_vals: dict[str, float] | None = None
    if grad:
        g0 = grad[0]
        g_vals = {
            "success_rate": 100.0
            * g0["successful_requests"]
            / max(g0["total_requests"], 1),
            "avg_latency_ms": g0["avg_completion_time"] * 1000,
            "throughput_rps": float(g0["throughput_requests_per_second"]),
            "tokens_per_sec": g0["avg_tokens_per_second"],
            "error_rate": g0["error_rate"],
            "cpu": g0["avg_cpu_usage"],
        }

    e_vals: dict[str, float] | None = None
    if endurance:
        e_vals = {
            "success_rate": 100.0
            * endurance["successful_requests"]
            / max(endurance["total_requests"], 1),
            "avg_latency_ms": endurance["avg_completion_time"] * 1000,
            "throughput_rps": float(endurance["throughput_requests_per_second"]),
            "tokens_per_sec": endurance["avg_tokens_per_second"],
            "error_rate": endurance["error_rate"],
            "cpu": endurance["avg_cpu_usage"],
        }

    if e_vals is not None:
        stress_vllm = {**e_vals}
    elif g_vals is not None:
        stress_vllm = {**g_vals}

    metric_keys = (
        "success_rate",
        "avg_latency_ms",
        "throughput_rps",
        "tokens_per_sec",
        "error_rate",
        "cpu",
    )
    if g_vals is not None and e_vals is not None:
        for k in metric_keys:
            stress_vllm[f"{k}_std"] = _std([float(g_vals[k]), float(e_vals[k])])
    elif stress_vllm:
        for k in metric_keys:
            stress_vllm[f"{k}_std"] = 0.0

    return {
        "scenario1": {
            "ollama": s1_metrics(s1["ollama"]),
            "vllm": s1_metrics(s1["vllm"]),
        },
        "scenario2": {"ollama": s2_ollama_row(), "vllm": s2_vllm_row()},
        "scenario3": {
            "ollama": s3_metrics(s3["ollama"]),
            "vllm": s3_metrics(s3["vllm"]),
        },
        "scenario4": {"vllm": stress_vllm},
    }

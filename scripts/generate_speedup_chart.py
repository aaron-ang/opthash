#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
CRITERION_DIR = ROOT / "target" / "criterion"
ASSETS_DIR = ROOT / "assets"

IMPLEMENTATIONS = ("std", "elastic", "funnel")

THROUGHPUT_WORKLOADS = (
    ("insert_throughput", "Insert"),
    ("get_hit_throughput", "Get Hit"),
    ("get_miss_throughput", "Get Miss"),
    ("mixed_lookup_throughput", "Mixed"),
    ("tiny_lookup_throughput", "Tiny"),
    ("delete_heavy_throughput", "Delete"),
    ("resize_heavy_throughput", "Resize"),
)

LATENCY_SIZES = ("100", "1K", "10K", "100K", "1M", "10M")


def load_mean_ns(workload: str, implementation: str) -> float:
    path = CRITERION_DIR / workload / implementation / "new" / "estimates.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Criterion estimates: {path}")
    data = json.loads(path.read_text())
    if "mean" in data and "point_estimate" in data["mean"]:
        return float(data["mean"]["point_estimate"])
    raise RuntimeError(f"no usable mean point estimate in {path}")


def plot_throughput_speedup(output_path: Path):
    """Single bar chart: all throughput workloads, speedup vs std."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    elastic_speedups = []
    funnel_speedups = []

    for workload, label in THROUGHPUT_WORKLOADS:
        try:
            times = {impl: load_mean_ns(workload, impl) for impl in IMPLEMENTATIONS}
        except FileNotFoundError:
            continue
        labels.append(label)
        elastic_speedups.append(times["std"] / times["elastic"])
        funnel_speedups.append(times["std"] / times["funnel"])

    if not labels:
        print("no throughput data found, skipping")
        return

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)

    x = np.arange(len(labels))
    w = 0.34

    elastic_bars = ax.bar(x - w / 2, elastic_speedups, width=w, label="ElasticHashMap")
    funnel_bars = ax.bar(x + w / 2, funnel_speedups, width=w, label="FunnelHashMap")

    max_val = max(1.0, *(elastic_speedups + funnel_speedups))
    ax.set_ylim(0.0, max_val * 1.30)
    ax.axhline(1.0, linestyle="--", color="0.4", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_ylabel("Speedup (higher is better)", fontsize=14)
    ax.set_xlabel("Workload", fontsize=14, labelpad=14)

    ax.set_title(
        "Throughput Speedup over std::HashMap",
        fontsize=22,
        pad=28,
        color="#2B2F36",
    )
    ax.text(
        0.5,
        1.02,
        "Criterion throughput benchmarks \u2014 std::HashMap is the 1.0\u00d7 baseline",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="0.35",
    )

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=12)

    for bars in (elastic_bars, funnel_bars):
        for bar in bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max_val * 0.03,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path.relative_to(ROOT)}")


def plot_latency(output_path: Path):
    """Line chart: per-lookup latency (ns) vs map size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sizes_found: list[str] = []
    data: dict[str, list[float]] = {impl: [] for impl in IMPLEMENTATIONS}

    for size_label in LATENCY_SIZES:
        group = f"get_hit_latency_{size_label}"
        try:
            times = {impl: load_mean_ns(group, impl) for impl in IMPLEMENTATIONS}
        except FileNotFoundError:
            continue
        sizes_found.append(size_label)
        for impl in IMPLEMENTATIONS:
            data[impl].append(times[impl])

    if not sizes_found:
        print("no latency data found, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    x = np.arange(len(sizes_found))
    markers = {"std": "o", "elastic": "s", "funnel": "D"}
    labels = {
        "std": "std::HashMap",
        "elastic": "ElasticHashMap",
        "funnel": "FunnelHashMap",
    }

    for impl in IMPLEMENTATIONS:
        ax.plot(
            x,
            data[impl],
            marker=markers[impl],
            linewidth=2,
            markersize=7,
            label=labels[impl],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sizes_found, fontsize=12)
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylabel("Latency per lookup (ns)", fontsize=14)
    ax.set_xlabel("Map size (entries)", fontsize=14, labelpad=14)

    ax.set_title(
        "Get-Hit Latency vs Map Size",
        fontsize=22,
        pad=28,
        color="#2B2F36",
    )
    ax.text(
        0.5,
        1.02,
        "Single get() call \u2014 lower is better",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="0.35",
    )

    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path.relative_to(ROOT)}")


def main():
    plot_throughput_speedup(ASSETS_DIR / "benchmark-speedup.svg")
    plot_latency(ASSETS_DIR / "benchmark-latency.svg")


if __name__ == "__main__":
    main()

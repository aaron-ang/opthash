#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
CRITERION_DIR = ROOT / "target" / "criterion"
ASSETS_DIR = ROOT / "assets"


class SpeedupRow(TypedDict):
    workload: str
    label: str
    elastic_speedup: float
    funnel_speedup: float
    std_throughput: float
    elastic_throughput: float
    funnel_throughput: float


class ChartConfig(TypedDict):
    title: str
    subtitle: str
    output: Path
    workloads: tuple[tuple[str, str, int], ...]


CHARTS: tuple[ChartConfig, ...] = (
    {
        "title": "Core Throughput Speedup over std::HashMap",
        "subtitle": "Criterion throughput benchmarks, std::HashMap is the 1.0x baseline",
        "output": ASSETS_DIR / "benchmark-speedup-core.svg",
        "workloads": (
            ("insert_throughput", "Insert", 10_000),
            ("get_hit_throughput", "Get Hit", 200_000),
            ("get_miss_throughput", "Get Miss", 20_000),
            ("mixed_lookup_throughput", "Mixed Lookup", 100_000),
        ),
    },
    {
        "title": "Secondary Throughput Speedup over std::HashMap",
        "subtitle": "Specialized Criterion workloads, std::HashMap is the 1.0x baseline",
        "output": ASSETS_DIR / "benchmark-speedup-secondary.svg",
        "workloads": (
            ("tiny_lookup_throughput", "Tiny Lookup", 20_000),
            ("delete_heavy_throughput", "Delete Heavy", 12_000),
            ("resize_heavy_throughput", "Resize Heavy", 8_000),
        ),
    },
)

IMPLEMENTATIONS = ("std", "elastic", "funnel")


def load_estimate_time_ns(workload: str, implementation: str):
    path = CRITERION_DIR / workload / implementation / "new" / "estimates.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Criterion estimates: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse JSON from {path}: {exc}") from exc

    if "slope" in data and "point_estimate" in data["slope"]:
        return float(data["slope"]["point_estimate"])
    if "mean" in data and "point_estimate" in data["mean"]:
        return float(data["mean"]["point_estimate"])
    raise RuntimeError(f"no usable slope.mean point estimate in {path}")


def throughput_from_time_ns(operation_count: int, time_ns: float):
    if time_ns <= 0:
        raise ValueError(f"non-positive benchmark time: {time_ns}")
    return operation_count / (time_ns / 1_000_000_000.0)


def build_speedup_rows(
    workloads: tuple[tuple[str, str, int], ...],
):
    rows: list[SpeedupRow] = []
    for workload, label, operation_count in workloads:
        times = {
            implementation: load_estimate_time_ns(workload, implementation)
            for implementation in IMPLEMENTATIONS
        }
        throughputs = {
            implementation: throughput_from_time_ns(
                operation_count, times[implementation]
            )
            for implementation in IMPLEMENTATIONS
        }

        rows.append(
            {
                "workload": workload,
                "label": label,
                "elastic_speedup": times["std"] / times["elastic"],
                "funnel_speedup": times["std"] / times["funnel"],
                "std_throughput": throughputs["std"],
                "elastic_throughput": throughputs["elastic"],
                "funnel_throughput": throughputs["funnel"],
            }
        )
    return rows


def plot_chart(rows: list[SpeedupRow], title: str, subtitle: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["label"]) for row in rows]
    elastic_speedups = [float(row["elastic_speedup"]) for row in rows]
    funnel_speedups = [float(row["funnel_speedup"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)

    x_positions = np.arange(len(labels))
    bar_width = 0.34

    elastic_bars = ax.bar(
        x_positions - bar_width / 2,
        elastic_speedups,
        width=bar_width,
        label="ElasticHashMap",
    )
    funnel_bars = ax.bar(
        x_positions + bar_width / 2,
        funnel_speedups,
        width=bar_width,
        label="FunnelHashMap",
    )

    max_speedup = max(1.0, *(elastic_speedups + funnel_speedups))
    ax.set_ylim(0.0, max_speedup * 1.35)
    ax.axhline(1.0, linestyle="--")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    ax.set_ylabel("Speed up, higher is better", fontsize=14)
    ax.set_xlabel("Workload", fontsize=14, labelpad=14)

    ax.set_title(title, fontsize=23, pad=28, color="#2B2F36")
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="0.35",
    )

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=12)

    for bars in (elastic_bars, funnel_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_speedup * 0.04,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
            )

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    for chart in CHARTS:
        rows = build_speedup_rows(chart["workloads"])
        plot_chart(rows, chart["title"], chart["subtitle"], chart["output"])
        print(f"wrote {chart['output'].relative_to(ROOT)}")


if __name__ == "__main__":
    main()

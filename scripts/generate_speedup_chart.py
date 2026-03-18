#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
CRITERION_DIR = ROOT / "target" / "criterion"
OUTPUT_PATH = ROOT / "assets" / "benchmark-speedup.svg"

WORKLOADS = [
    ("insert_throughput", "Insert", 10_000),
    ("get_hit_throughput", "Get Hit", 200_000),
    ("get_miss_throughput", "Get Miss", 20_000),
    ("mixed_lookup_throughput", "Mixed Lookup", 100_000),
]
IMPLEMENTATIONS = ("std", "elastic", "funnel")


def load_estimate_time_ns(workload: str, implementation: str) -> float:
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


def throughput_from_time_ns(operation_count: int, time_ns: float) -> float:
    if time_ns <= 0:
        raise ValueError(f"non-positive benchmark time: {time_ns}")
    return operation_count / (time_ns / 1_000_000_000.0)


def build_speedup_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for workload, label, operation_count in WORKLOADS:
        times = {
            implementation: load_estimate_time_ns(workload, implementation)
            for implementation in IMPLEMENTATIONS
        }
        throughputs = {
            implementation: throughput_from_time_ns(operation_count, times[implementation])
            for implementation in IMPLEMENTATIONS
        }

        rows.append(
            {
                "workload": workload,
                "label": label,
                "std_speedup": 1.0,
                "elastic_speedup": times["std"] / times["elastic"],
                "funnel_speedup": times["std"] / times["funnel"],
                "std_throughput": throughputs["std"],
                "elastic_throughput": throughputs["elastic"],
                "funnel_throughput": throughputs["funnel"],
            }
        )
    return rows


def plot_chart(rows: list[dict[str, float | str]]) -> None:
    output_dir = OUTPUT_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(row["label"]) for row in rows]
    elastic_speedups = [float(row["elastic_speedup"]) for row in rows]
    funnel_speedups = [float(row["funnel_speedup"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_positions = list(range(len(labels)))
    bar_width = 0.34
    elastic_color = "#139A43"
    funnel_color = "#2D6CDF"

    elastic_bars = ax.bar(
        [x - bar_width / 2 for x in x_positions],
        elastic_speedups,
        width=bar_width,
        color=elastic_color,
        label="ElasticHashMap",
        zorder=3,
    )
    funnel_bars = ax.bar(
        [x + bar_width / 2 for x in x_positions],
        funnel_speedups,
        width=bar_width,
        color=funnel_color,
        label="FunnelHashMap",
        zorder=3,
    )

    max_speedup = max(1.0, *(elastic_speedups + funnel_speedups))
    ax.set_ylim(0.0, max_speedup * 1.35)
    ax.yaxis.grid(True, color="#D7DCE5", linewidth=1.0, zorder=0)
    ax.xaxis.grid(False)
    ax.axhline(1.0, color="#6B7280", linewidth=1.4, linestyle="--", zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    ax.set_ylabel("Speed up, higher is better", fontsize=14)

    ax.set_title(
        "Throughput Speedup of HashMap over std::HashMap",
        fontsize=23,
        pad=28,
        color="#2B2F36",
    )
    ax.text(
        0.5,
        1.02,
        "Criterion throughput benchmarks, std::HashMap is the 1.0x baseline",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color="#6B7280",
    )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    for text in legend.get_texts():
        text.set_color("#2B2F36")

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
                color=bar.get_facecolor(),
                fontweight="bold",
            )

    fig.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = build_speedup_rows()
    plot_chart(rows)
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

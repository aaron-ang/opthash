from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    ASSETS_DIR,
    IMPLEMENTATIONS,
    apply_axis_style,
    load_criterion_mean_ns,
    save_svg,
)


THROUGHPUT_WORKLOADS = (
    ("insert_throughput", "Insert"),
    ("get_hit_throughput", "Get Hit"),
    ("get_miss_throughput", "Get Miss"),
    ("mixed_lookup_throughput", "Mixed"),
    ("tiny_lookup_throughput", "Tiny"),
    ("delete_heavy_throughput", "Delete"),
    ("resize_heavy_throughput", "Resize"),
)


def plot_throughput_speedup(output_path: Path) -> None:
    """Single bar chart: all throughput workloads, speedup vs std."""
    labels = []
    elastic_speedups = []
    funnel_speedups = []

    for workload, label in THROUGHPUT_WORKLOADS:
        try:
            times = {
                impl: load_criterion_mean_ns(workload, impl) for impl in IMPLEMENTATIONS
            }
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
    apply_axis_style(
        ax,
        title="Throughput Speedup over std::HashMap",
        subtitle="Criterion throughput benchmarks \u2014 std::HashMap is the 1.0\u00d7 baseline",
        xlabel="Workload",
        ylabel="Speedup (higher is better)",
        y_formatter=lambda v, _: f"{v:.1f}",
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

    save_svg(fig, output_path)


def main() -> None:
    plot_throughput_speedup(ASSETS_DIR / "benchmark-speedup.svg")


if __name__ == "__main__":
    main()

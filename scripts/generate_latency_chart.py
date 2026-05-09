from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

from plot_common import (
    ASSETS_DIR,
    IMPLEMENTATIONS,
    IMPL_LABELS,
    IMPL_MARKERS,
    LATENCY_SIZES,
    apply_axis_style,
    load_criterion_mean_ns,
    load_latency_json,
    save_svg,
)


OPS = ("get-hit", "get-miss", "insert")
OP_LABELS = {"get-hit": "Get Hit", "get-miss": "Get Miss", "insert": "Insert"}


def plot_mean_latency_by_size(output_path: Path) -> None:
    """Criterion-mean per-lookup latency vs map size. Reads target/criterion/."""
    sizes_found: list[str] = []
    data: dict[str, list[float]] = {impl: [] for impl in IMPLEMENTATIONS}

    for size_label in LATENCY_SIZES:
        group = f"get_hit_latency_{size_label}"
        try:
            times = {
                impl: load_criterion_mean_ns(group, impl) for impl in IMPLEMENTATIONS
            }
        except FileNotFoundError:
            continue
        sizes_found.append(size_label)
        for impl in IMPLEMENTATIONS:
            data[impl].append(times[impl])

    if not sizes_found:
        print("no criterion latency data found, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    x = np.arange(len(sizes_found))

    for impl in IMPLEMENTATIONS:
        ax.plot(
            x,
            data[impl],
            marker=IMPL_MARKERS[impl],
            linewidth=2,
            markersize=7,
            label=IMPL_LABELS[impl],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sizes_found, fontsize=12)
    apply_axis_style(
        ax,
        title="Get-Hit Latency vs Map Size",
        subtitle="Single get() call \u2014 lower is better",
        xlabel="Map size (entries)",
        ylabel="Latency per lookup (ns)",
        y_formatter=lambda v, _: f"{v:.0f}",
    )

    ax.legend(fontsize=12)

    save_svg(fig, output_path)


def _percentile_curve(
    buckets: list[dict], overhead_ns: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (quantile, latency_ns) from sorted-by-ns_high bucket list."""
    highs = np.array([b["ns_high"] for b in buckets], dtype=float)
    highs = np.maximum(highs - overhead_ns, 1.0)
    counts = np.array([b["count"] for b in buckets], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.array([]), np.array([])
    cum_q = np.cumsum(counts) / total
    return cum_q, highs


TAIL_TICK_QS = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)
TAIL_TICK_LABELS = ("p0", "p50", "p90", "p99", "p99.9", "p99.99", "p99.999")


def _tail_x(q):
    """Map quantile q in [0, 1) onto a log-scaled tail axis: x = 1 / (1 - q)."""
    return 1.0 / np.maximum(1.0 - np.asarray(q, dtype=float), 1e-7)


def plot_tail_cdf(size: int, op: str, output_path: Path) -> None:
    """Percentile-vs-latency tail plot for one (size, op) config, three lines."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    any_data = False
    max_x = _tail_x(TAIL_TICK_QS[-1])
    for impl in IMPLEMENTATIONS:
        doc = load_latency_json(impl, size, op)
        if doc is None:
            continue
        buckets = doc.get("histogram", [])
        if not buckets:
            continue
        overhead = float(doc.get("clock_overhead_ns", 0))
        q, y = _percentile_curve(buckets, overhead)
        if q.size == 0:
            continue
        any_data = True
        x = _tail_x(q)
        ax.plot(
            x,
            y,
            marker=IMPL_MARKERS[impl],
            markersize=4,
            markevery=max(1, len(x) // 20),
            linewidth=2,
            label=IMPL_LABELS[impl],
        )

    if not any_data:
        plt.close(fig)
        print(f"no latency data for size={size} op={op}, skipping tail plot")
        return

    tick_positions = [_tail_x(q) for q in TAIL_TICK_QS]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(TAIL_TICK_LABELS, fontsize=12)
    ax.set_xlim(1.0, max_x * 1.3)

    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    size_label = f"{size:,}"
    apply_axis_style(
        ax,
        title=f"Tail Latency \u2014 {OP_LABELS[op]} @ {size_label} entries",
        subtitle="Latency at percentile p (log axes) \u2014 lower is better",
        xlabel="Percentile",
        ylabel="Latency (ns, log scale)",
    )
    ax.legend(fontsize=12, loc="upper left")

    save_svg(fig, output_path)


TAIL_CONFIGS: tuple[tuple[int, str], ...] = (
    (1_000_000, "get-hit"),
    (1_000_000, "get-miss"),
    (1_000_000, "insert"),
)


def plot_all_tail_charts(assets_dir: Path) -> None:
    for size, op in TAIL_CONFIGS:
        plot_tail_cdf(size, op, assets_dir / f"latency-tail-{size}-{op}.svg")


def main() -> None:
    plot_mean_latency_by_size(ASSETS_DIR / "benchmark-latency.svg")
    plot_all_tail_charts(ASSETS_DIR)


if __name__ == "__main__":
    main()

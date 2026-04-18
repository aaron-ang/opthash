from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    ASSETS_DIR,
    IMPLEMENTATIONS,
    IMPL_LABELS,
    IMPL_MARKERS,
    LATENCY_DIR,
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
            times = {impl: load_criterion_mean_ns(group, impl) for impl in IMPLEMENTATIONS}
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


def _percentile_curve(buckets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (quantile, latency_ns) from sorted-by-ns_high bucket list."""
    highs = np.array([b["ns_high"] for b in buckets], dtype=float)
    counts = np.array([b["count"] for b in buckets], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.array([]), np.array([])
    cum_q = np.cumsum(counts) / total
    return cum_q, highs


TAIL_TICK_QS = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)
TAIL_TICK_LABELS = ("0", "0.5", "0.9", "0.99", "0.999", "0.9999", "0.99999")


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
        q, y = _percentile_curve(buckets)
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

    size_label = f"{size:,}".replace(",", "\u202f")
    apply_axis_style(
        ax,
        title=f"Tail Latency \u2014 {OP_LABELS[op]} @ {size_label} entries",
        subtitle="Latency at percentile q (log axes) \u2014 lower is better",
        xlabel="Percentile (q)",
        ylabel="Latency (ns, log scale)",
        y_formatter=lambda v, _: f"{v:,.0f}" if v >= 1 else "",
    )
    ax.legend(fontsize=12, loc="upper left")

    save_svg(fig, output_path)


def discover_configs() -> list[tuple[int, str]]:
    """Return (size, op) pairs present under target/latency/ for at least one impl."""
    found: set[tuple[int, str]] = set()
    if not LATENCY_DIR.exists():
        return []
    for impl_dir in LATENCY_DIR.iterdir():
        if not impl_dir.is_dir():
            continue
        for size_dir in impl_dir.iterdir():
            if not size_dir.is_dir():
                continue
            try:
                size = int(size_dir.name)
            except ValueError:
                continue
            for op_file in size_dir.glob("*.json"):
                op = op_file.stem
                if op in OPS:
                    found.add((size, op))
    return sorted(found)


def plot_all_tail_charts(assets_dir: Path) -> None:
    configs = discover_configs()
    if not configs:
        print("no tail latency data found under target/latency/, skipping")
        return
    for size, op in configs:
        plot_tail_cdf(size, op, assets_dir / f"latency-tail-{size}-{op}.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latency charts")
    parser.add_argument("--size", type=int, help="Map size to plot (omit to plot all found)")
    parser.add_argument("--op", choices=OPS, help="Operation to plot (omit to plot all found)")
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Only regenerate the Criterion-mean chart",
    )
    args = parser.parse_args()

    plot_mean_latency_by_size(ASSETS_DIR / "benchmark-latency.svg")
    if args.mean_only:
        return

    if args.size is not None and args.op is not None:
        plot_tail_cdf(args.size, args.op, ASSETS_DIR / f"latency-tail-{args.size}-{args.op}.svg")
        return

    configs = discover_configs()
    if args.size is not None:
        configs = [(s, o) for s, o in configs if s == args.size]
    if args.op is not None:
        configs = [(s, o) for s, o in configs if o == args.op]
    if not configs:
        print("no matching tail latency data, skipping")
        return
    for size, op in configs:
        plot_tail_cdf(size, op, ASSETS_DIR / f"latency-tail-{size}-{op}.svg")


if __name__ == "__main__":
    main()

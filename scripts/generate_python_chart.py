import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import ASSETS_DIR, ROOT, apply_axis_style, save_svg


BENCHMARKS_JSON = ROOT / ".benchmarks" / "python.json"
OUTPUT = ASSETS_DIR / "benchmark-python-speedup.svg"

WORKLOADS = (
    ("insert", "Insert"),
    ("get_hit", "Get Hit"),
    ("get_miss", "Get Miss"),
    ("mixed", "Mixed"),
    ("delete", "Delete"),
)

IMPLS = ("dict", "elastic", "funnel")
PY_IMPL_LABELS = {
    "dict": "builtin dict",
    "elastic": "ElasticHashMap",
    "funnel": "FunnelHashMap",
}

NAME_RE = re.compile(r"\[(\w+)\]$")


def load_means(path: Path) -> dict[str, dict[str, float]]:
    """Return {group: {impl: mean_seconds}} from pytest-benchmark JSON."""
    if not path.exists():
        sys.exit(
            f"missing {path.relative_to(ROOT)}\n"
            "  run: pytest benches/ --benchmark-json=.benchmarks/python.json"
        )
    data = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for b in data["benchmarks"]:
        group = b["group"]
        m = NAME_RE.search(b["name"])
        if not m:
            continue
        impl = m.group(1)
        out.setdefault(group, {})[impl] = float(b["stats"]["mean"])
    return out


def plot_speedup(means: dict[str, dict[str, float]], output_path: Path) -> None:
    labels: list[str] = []
    elastic_speedups: list[float] = []
    funnel_speedups: list[float] = []

    for key, label in WORKLOADS:
        entry = means.get(key)
        if not entry or not all(impl in entry for impl in IMPLS):
            continue
        labels.append(label)
        elastic_speedups.append(entry["dict"] / entry["elastic"])
        funnel_speedups.append(entry["dict"] / entry["funnel"])

    if not labels:
        sys.exit("no workloads found in benchmark JSON")

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    x = np.arange(len(labels))
    w = 0.34

    elastic_bars = ax.bar(
        x - w / 2, elastic_speedups, width=w, label=PY_IMPL_LABELS["elastic"]
    )
    funnel_bars = ax.bar(
        x + w / 2, funnel_speedups, width=w, label=PY_IMPL_LABELS["funnel"]
    )

    max_val = max(1.0, *(elastic_speedups + funnel_speedups))
    ax.set_ylim(0.0, max_val * 1.30)
    ax.axhline(1.0, linestyle="--", color="0.4", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    apply_axis_style(
        ax,
        title=f"Python-side Throughput Speedup over {PY_IMPL_LABELS['dict']}",
        subtitle=f"pytest-benchmark — {PY_IMPL_LABELS['dict']} is the 1.0× baseline",
        xlabel="Workload",
        ylabel="Speedup (higher is better)",
        y_formatter=lambda v, _: f"{v:.2f}",
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
    means = load_means(BENCHMARKS_JSON)
    plot_speedup(means, OUTPUT)


if __name__ == "__main__":
    main()

from __future__ import annotations

from generate_latency_chart import plot_all_tail_charts, plot_mean_latency_by_size
from generate_speedup_chart import plot_throughput_speedup
from plot_common import ASSETS_DIR


def main() -> None:
    plot_throughput_speedup(ASSETS_DIR / "benchmark-speedup.svg")
    plot_mean_latency_by_size(ASSETS_DIR / "benchmark-latency.svg")
    plot_all_tail_charts(ASSETS_DIR)


if __name__ == "__main__":
    main()

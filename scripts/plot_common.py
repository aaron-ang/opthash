from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
CRITERION_DIR = ROOT / "target" / "criterion"
LATENCY_DIR = ROOT / "target" / "latency"
ASSETS_DIR = ROOT / "assets"

IMPLEMENTATIONS = ("std", "elastic", "funnel")
IMPL_LABELS = {
    "std": "std::HashMap",
    "elastic": "ElasticHashMap",
    "funnel": "FunnelHashMap",
}
IMPL_MARKERS = {"std": "o", "elastic": "s", "funnel": "D"}

LATENCY_SIZES = ("100", "1K", "10K", "100K", "1M", "10M")

TITLE_COLOR = "#2B2F36"
SUBTITLE_COLOR = "0.35"


def load_criterion_mean_ns(group: str, implementation: str) -> float:
    path = CRITERION_DIR / group / implementation / "new" / "estimates.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Criterion estimates: {path}")
    data = json.loads(path.read_text())
    if "mean" in data and "point_estimate" in data["mean"]:
        return float(data["mean"]["point_estimate"])
    raise RuntimeError(f"no usable mean point estimate in {path}")


def load_latency_json(implementation: str, size: int, op: str) -> dict | None:
    path = LATENCY_DIR / implementation / str(size) / f"{op}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def apply_axis_style(
    ax,
    *,
    title: str,
    subtitle: str | None = None,
    xlabel: str,
    ylabel: str,
    y_formatter=None,
) -> None:
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.tick_params(axis="x", length=0)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=14)
    ax.set_title(title, fontsize=22, pad=28, color=TITLE_COLOR)
    if subtitle is not None:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            color=SUBTITLE_COLOR,
        )


def save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")

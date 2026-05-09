# AGENTS.md

Run all of these after every refactor. Check benchmark results in `target/criterion/` for performance regressions.

## Commands

```bash
cargo fmt                                   # Format Rust code
cargo clippy --all-features -- -W clippy::pedantic   # Lint with pedantic warnings
cargo test                                  # Run all tests
cargo bench                                 # Run all benchmarks
uvx ruff format                             # Format Python code (scripts/, tests/)
pre-commit run --all-files                  # Run formatters on the whole tree
```

One-time setup (after cloning):

```bash
uv tool install pre-commit
pre-commit install
```

## Benchmarks

Criterion benchmark suite comparing `ElasticHashMap`, `FunnelHashMap`, `std::HashMap`, and `hashbrown::HashMap` (SwissTable + foldhash — absolute ceiling reference):

- **`cargo bench --bench speedup`** — throughput (insert, get hit/miss/mixed/tiny, delete-heavy, resize-heavy) and Criterion-mean per-lookup latency at varying map sizes
- Run a subset: `cargo bench --bench speedup -- "get_hit_latency"` (Criterion name filter)

Criterion auto-compares against the previous run. **Always read results from the JSON files** — terminal output gets truncated and mixes runs. Parse `target/criterion/` after a single `cargo bench` invocation:

- `target/criterion/<group>/<variant>/new/estimates.json` — absolute timing (`mean.point_estimate` in nanoseconds)
- `target/criterion/<group>/<variant>/change/estimates.json` — relative change vs. previous run (`mean.point_estimate` as a fraction, e.g. +0.05 = 5% slower)

Example path: `target/criterion/get_hit_throughput/elastic/change/estimates.json`

### Tail-latency harness

- **`cargo bench --bench latency`** — per-operation latency distributions (p50/p90/p99/p999/p9999/max) via `hdrhistogram`. Hard-coded matrix (edit the consts at the top of `benches/latency.rs` to change): sizes 10K/100K/1M/10M × ops get-hit/get-miss/insert × all four maps × 1M samples × 10K warmup. `insert` pre-fills to `size − samples` before measuring so tail percentiles reflect insert cost near the target load, not into an empty map.
- Output: `target/latency/<map>/<size>/<op>.json` — percentiles + histogram buckets + `clock_overhead_ns`.

### Python-side benchmarks

`benches/test_python.py` — pytest-benchmark suite comparing `dict`, `ElasticHashMap`, and `FunnelHashMap` from Python across insert / get_hit / get_miss / mixed / delete workloads at N = 10K. Each opthash op crosses the GIL → `HashedAny::hash()` → Python bytecode.

```bash
pytest benches/test_python.py --benchmark-json=.benchmarks/python.json

uv run --group charts python scripts/generate_python_chart.py
```

### Charts

- `uv run --group charts scripts/generate_speedup_chart.py` — throughput speedup bar chart
- `uv run --group charts scripts/generate_latency_chart.py` — Criterion-mean latency line + pinned tail CDFs (edit `TAIL_CONFIGS` in the script to re-pin; currently 1M × 3 ops)
- `uv run --group charts scripts/generate_all_charts.py` — regenerate everything
- `uv run --group charts scripts/generate_python_chart.py` — Python-side dict-vs-opthash speedup (reads `.benchmarks/python.json`)

Charts are saved in `assets/`. Shared plotting helpers (`IMPLEMENTATIONS`, loaders, axis styling) live in `scripts/plot_common.py`. The tail plotter subtracts `clock_overhead_ns` so percentiles reflect per-op latency, not per-(op + `Instant::now()`).

## Project structure

- `src/elastic.rs` — `ElasticHashMap` (tests inline)
- `src/funnel.rs` — `FunnelHashMap` (tests inline)
- `src/common/` — shared internals: control-byte SIMD ops, layout math, config

## Refactoring guidelines

- Low-level helpers used by both the library and benchmarks live in `src/common/` (bitmask, simd, layout, math). Benches pull fixtures from `benches/common.rs`. Don't duplicate primitives across `src/` and `benches/`.
- Prefer layout and locality wins before adding more metadata.
- Keep hot metadata contiguous. If fields are read together, store them together.
- Avoid metadata that is expensive to maintain on every insert or delete unless benchmarks prove it wins overall.
- Cache routing state that is reused in hot paths. Do not recompute it per probe.
- Preserve SIMD-friendly control-byte scans: contiguous groups, cheap bitmask iteration, and early rejection before touching payloads.
- Reject optimizations that improve only microbenchmarks but regress the public `throughput` suite.
- Profile hot functions before and after changes. In this repo, focus on `find_slot_indices_with_hash` / `find_in_level_by_probe` (elastic), `find_slot_location_with_hash` / `find_in_level_bucket` (funnel), `group_probe_params`, `choose_slot_for_new_key`, and the resize paths.
- Use `target/criterion/` as the final gate. If the relevant benchmark regresses, the optimization does not stay.

# AGENTS.md

Run all of these after every refactor. Check benchmark results in `target/criterion/` for performance regressions.

## Commands

```bash
cargo fmt                                   # Format code
cargo clippy -- -W clippy::pedantic         # Lint with pedantic warnings
cargo test                                  # Run all tests
cargo bench                                 # Run all benchmarks
```

## Benchmarks

Criterion benchmark suite comparing `ElasticHashMap`, `FunnelHashMap`, and `std::HashMap`:

- **`cargo bench --bench speedup`** — throughput (insert, get hit/miss/mixed/tiny, delete-heavy, resize-heavy) and Criterion-mean per-lookup latency at varying map sizes
- Run a subset: `cargo bench --bench speedup -- "get_hit_latency"` (Criterion name filter)

Criterion auto-compares against the previous run. **Always read results from the JSON files** — terminal output gets truncated and mixes runs. Parse `target/criterion/` after a single `cargo bench` invocation:

- `target/criterion/<group>/<variant>/new/estimates.json` — absolute timing (`mean.point_estimate` in nanoseconds)
- `target/criterion/<group>/<variant>/change/estimates.json` — relative change vs. previous run (`mean.point_estimate` as a fraction, e.g. +0.05 = 5% slower)

Example path: `target/criterion/get_hit_throughput/elastic/change/estimates.json`

### Tail-latency harness

- **`cargo bench --bench latency`** — per-operation latency distributions (p50/p90/p99/p999/p9999/max) via `hdrhistogram`. Defaults sweep sizes 10K/100K/1M × ops get-hit/get-miss/insert × all three maps.
- CLI filters: `cargo bench --bench latency -- --size 100000 --op get-hit --map elastic --samples 500000 --warmup 10000` (comma-separate to pass multiple).
- Output: `target/latency/<map>/<size>/<op>.json` — percentiles + histogram buckets.

### Charts

- `uv run scripts/generate_speedup_chart.py` — throughput speedup bar chart
- `uv run scripts/generate_latency_chart.py [--size N --op OP] [--mean-only]` — Criterion-mean latency line + per-config tail CDFs + percentile bars
- `uv run scripts/generate_all_charts.py` — regenerate everything

Charts are saved in `assets/`. Shared plotting helpers live in `scripts/plot_common.py`.

## Project structure

- `src/elastic.rs` — `ElasticHashMap` (tests inline)
- `src/funnel.rs` — `FunnelHashMap` (tests inline)
- `src/common/` — shared internals: control-byte SIMD ops, layout math, config

## Refactoring guidelines

- If a low-level helper is used by both the root crate and benchmarks, move it into `opthash-internal/` instead of duplicating it or exposing bench-only API from `src/`.
- Prefer layout and locality wins before adding more metadata.
- Keep hot metadata contiguous. If fields are read together, store them together.
- Avoid metadata that is expensive to maintain on every insert or delete unless benchmarks prove it wins overall.
- Cache routing state that is reused in hot paths. Do not recompute it per probe.
- Preserve SIMD-friendly control-byte scans: contiguous groups, cheap bitmask iteration, and early rejection before touching payloads.
- Reject optimizations that improve only microbenchmarks but regress the public `throughput` suite.
- Profile hot functions before and after changes. In this repo, focus on `find_*`, `first_free_*`, `place_new_entry`, and constructor/resize paths.
- Use `target/criterion/` as the final gate. If the relevant benchmark regresses, the optimization does not stay.

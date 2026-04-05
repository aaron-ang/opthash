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

- **`cargo bench --bench benchmarks`** — throughput (insert, get hit/miss/mixed/tiny, delete-heavy, resize-heavy) and per-lookup latency at varying map sizes
- Run a subset: `cargo bench --bench benchmarks -- "get_hit_latency"` (Criterion name filter)

Criterion auto-compares against the previous run. **Always read results from the JSON files** — terminal output gets truncated and mixes runs. Parse `target/criterion/` after a single `cargo bench` invocation:

- `target/criterion/<group>/<variant>/new/estimates.json` — absolute timing (`mean.point_estimate` in nanoseconds)
- `target/criterion/<group>/<variant>/change/estimates.json` — relative change vs. previous run (`mean.point_estimate` as a fraction, e.g. +0.05 = 5% slower)

Example path: `target/criterion/get_hit_throughput/elastic/change/estimates.json`

To generate speedup charts against `std::HashMap`, run `uv run scripts/generate_speedup_chart.py`. Charts are saved in `assets/`.

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

# AGENTS.md

Run all of these after every refactor. Check benchmark results in `target/criterion/` for performance regressions.

## Commands

```bash
cargo fmt --check                           # Check formatting
cargo clippy -- -W clippy::pedantic         # Lint with pedantic warnings
cargo test                                  # Run all tests
cargo bench                                 # Run all benchmarks
```

## Benchmarks

Two Criterion benchmark suites compare `ElasticHashMap`, `FunnelHashMap`, and `std::HashMap`:

- **`cargo bench --bench throughput`** — end-to-end operations: insert, get (hit/miss/mixed/tiny), delete-heavy, resize-heavy
- **`cargo bench --bench internal_paths`** — hot-path internals: control-byte scanning, miss-path rejection, bucket lookup with tombstones, delete churn, resize cost

Criterion auto-compares against the previous run. To check for regressions, read the JSON files under `target/criterion/`:

- `target/criterion/<group>/<variant>/new/estimates.json` — absolute timing (`mean.point_estimate` in nanoseconds)
- `target/criterion/<group>/<variant>/change/estimates.json` — relative change vs. previous run (`mean.point_estimate` as a fraction, e.g. +0.05 = 5% slower)

Example path: `target/criterion/get_hit_throughput/elastic/change/estimates.json`

## Project structure

- `src/elastic.rs` — `ElasticHashMap` (tests inline)
- `src/funnel.rs` — `FunnelHashMap` (tests inline)
- `src/common/` — shared internals: control-byte SIMD ops, layout math, config

## Refactoring guidelines

- If a low-level helper is used by both the root crate and benchmarks, move it into `opthash-internal/` instead of duplicating it or exposing bench-only API from `src/`.

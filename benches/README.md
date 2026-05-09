# Benchmarking

Rust bench targets compare `std::collections::HashMap`, `hashbrown::HashMap`, `opthash::ElasticHashMap`, `opthash::FunnelHashMap`. Shared fixtures live in `benches/common.rs`. A Python-side bench (`benches/python.py`) compares the opthash bindings against builtin `dict`.

## Results

### Throughput (Rust, vs `std::HashMap`)

![Throughput speedup chart](../assets/benchmark-speedup.svg)

### Mean latency by map size (Rust)

![Latency chart](../assets/benchmark-latency.svg)

### Tail latency distributions (Rust)

![Tail latency — get-hit](../assets/latency-tail-1000000-get-hit.svg)

![Tail latency — get-miss](../assets/latency-tail-1000000-get-miss.svg)

![Tail latency — insert](../assets/latency-tail-1000000-insert.svg)

### Python: opthash bindings vs builtin `dict`

![Python speedup chart](../assets/benchmark-python-speedup.svg)

## `benches/speedup.rs` — throughput + mean latency (Criterion)

Throughput workloads:

1. `insert_throughput`
2. `get_hit_throughput`
3. `get_miss_throughput`
4. `tiny_lookup_throughput`
5. `delete_heavy_throughput`
6. `resize_heavy_throughput`
7. `mixed_lookup_throughput`

The tiny-map workload exercises the internal tiny-table engine. Delete-heavy and resize-heavy expose tombstone handling and growth costs instead of only steady-state inserts.

Latency workload: `get_hit_latency_<size>` — Criterion-mean per-lookup time across map sizes (configured at the top of `benches/speedup.rs`).

Run:

```bash
cargo bench --bench speedup
cargo bench --bench speedup -- "get_hit"          # Criterion name filter
```

## `benches/latency.rs` — tail-latency histograms (hdrhistogram)

Captures per-operation latency distributions (p50/p90/p99/p999/p9999/max) and dumps them to JSON for plotting. Custom `harness = false` main, not Criterion. The size × op × map matrix and sample/warmup counts are hardcoded — edit the consts at the top of `benches/latency.rs` to change. Output: `target/latency/<map>/<size>/<op>.json`.

```bash
cargo bench --bench latency
```

## Reports

- Criterion HTML: `target/criterion/report/index.html`, per-workload pages below (e.g. `target/criterion/insert_throughput/report/index.html`)
- Charts: `uv run scripts/generate_all_charts.py` writes every SVG to `assets/` (speedup bars, mean-latency line, tail CDFs per config)

## Profiling / flamegraphs

`benches/speedup.rs` integrates a `pprof` profiler. Pass `--profile-time N` and Criterion captures CPU samples instead of timing, writing `target/criterion/<workload>/<impl>/profile/flamegraph.svg`.

```bash
cargo bench --bench speedup -- --profile-time 5
cargo bench --bench speedup -- --profile-time 5 "get_hit"
```

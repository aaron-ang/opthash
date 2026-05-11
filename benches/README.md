# Benchmarking

Rust bench targets compare `std::collections::HashMap`, `hashbrown::HashMap`, `opthash::ElasticHashMap`, `opthash::FunnelHashMap`. Shared fixtures live in `benches/common.rs`. A Python-side bench (`benches/test_python.py`) compares the opthash bindings against builtin `dict`.

## Results

### Throughput (Rust, vs `std::HashMap`)

![Throughput speedup chart](../assets/benchmark-speedup.svg)

### Mean latency by map size (Rust)

![Latency chart](../assets/benchmark-latency.svg)

### Tail latency distribution (Rust)

![Tail latency — get-hit @ 10M](../assets/latency-tail-10M-get-hit.svg)

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

## `benches/test_python.py` — Python bindings vs builtin `dict` (pytest-benchmark)

End-to-end workloads (insert / get_hit / get_miss / mixed / delete). Each opthash op crosses the GIL → `HashedAny::hash()` → Python bytecode, so this measures binding overhead as well as the map.

```bash
pytest benches/test_python.py --benchmark-json=.benchmarks/python.json
uv run --group charts python scripts/generate_python_chart.py
```

## `benches/profile_bindings.py` — per-op binding overhead

Decomposes one `m[k]` call: `loop -> hash(k) -> dict[k] -> __contains__ -> __getitem__ -> .get()`. Δ between rows attributes each primitive's ns cost. Run with `python benches/profile_bindings.py`.

For symbol-level attribution, drive a hot loop under `py-spy --native` and aggregate the folded-stack output:

```bash
py-spy record --native --rate 1000 --duration 8 \
  --format raw --output /tmp/perf_raw.txt -- \
  python benches/profile_bindings.py
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

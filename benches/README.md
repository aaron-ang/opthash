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

## `benches/python.py` — Python bindings vs builtin `dict` (pytest-benchmark)

End-to-end workloads (insert / get_hit / get_miss / mixed / delete) at N=10K. Each opthash op crosses the GIL → `HashedAny::hash()` → Python bytecode, so this measures binding overhead as well as the map.

```bash
pytest -o python_files='*.py' benches/python.py --benchmark-json=.benchmarks/python.json
uv run --group charts python scripts/generate_python_chart.py
```

The `python_files` override is needed because the file is named `python.py` to match the Rust naming convention (`speedup.rs`, `latency.rs`) rather than pytest's default `test_*.py` pattern.

## `benches/profile_bindings.py` — per-op decomposition microbench

Decomposes one `m[k]` call by comparing primitives in a tight `time.perf_counter_ns` loop:

```
loop only  ->  hash(k)  ->  dict[k]  ->  opthash __contains__  ->  __getitem__  ->  .get()
```

Δ between adjacent rows isolates the cost of one extra primitive on the path. Useful for attributing binding overhead (HashedAny construction, refcount churn, pyo3 dispatch) separately from map probe cost.

```bash
python benches/profile_bindings.py
```

Pair with native sampling for symbol-level attribution:

```bash
# Build a long-running hot loop, then sample with py-spy --native.
py-spy record --native --rate 1000 --duration 8 \
  --output /tmp/perf_raw.txt --format raw -- python -c '
import opthash
N=10_000; m=opthash.FunnelHashMap(capacity=N)
for i in range(N): m[f"key_{i}"]=0
for _ in range(5_000):
    for i in range(N): m.__contains__(f"key_{i}")
'
```

Aggregate inclusive frame counts from the folded-stack output to see which Rust symbols dominate (`HashedAny::eq`, `find_slot_*`, atomic refcount ops, pyo3 trampoline, etc.).

## Reports

- Criterion HTML: `target/criterion/report/index.html`, per-workload pages below (e.g. `target/criterion/insert_throughput/report/index.html`)
- Charts: `uv run scripts/generate_all_charts.py` writes every SVG to `assets/` (speedup bars, mean-latency line, tail CDFs per config)

## Profiling / flamegraphs

`benches/speedup.rs` integrates a `pprof` profiler. Pass `--profile-time N` and Criterion captures CPU samples instead of timing, writing `target/criterion/<workload>/<impl>/profile/flamegraph.svg`.

```bash
cargo bench --bench speedup -- --profile-time 5
cargo bench --bench speedup -- --profile-time 5 "get_hit"
```

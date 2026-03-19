# opthash

Rust implementations of **Elastic Hashing** and **Funnel Hashing** from [*Optimal Bounds for Open Addressing Without Reordering*](https://arxiv.org/abs/2501.02305) (Farach-Colton, Krapivin, Kuszmaul, 2025).

Both are open-addressing hash maps that achieve optimal expected probe complexity without reordering elements after insertion.

## Data Structures

- **`ElasticHashMap<K, V>`** — Multi-level table with geometrically halving levels. Keys are placed via batch-based insertion across levels using stride-based probing.
- **`FunnelHashMap<K, V>`** — Multi-level bucketed table with a 3/4-ratio geometric progression and a special overflow array (primary + fallback) for keys that don't fit in any level.

Both support `insert`, `get`, `get_mut`, `contains_key`, `remove`, and `clear`. Maps start with zero allocation (`new()`) and grow dynamically on demand. Advanced tuning is available through `ElasticOptions`, `FunnelOptions`, and `with_options(...)`.

## Benchmarks

Current Criterion throughput results on Apple M1 (aarch64, NEON SIMD), normalized so `std::HashMap` is the `1.0x` baseline:

![Benchmark speedup chart](assets/benchmark-speedup.svg)

Regenerate the benchmark chart:

```bash
cargo bench --bench throughput
uv venv
uv pip install -r requirements.txt
uv run scripts/generate_speedup_chart.py
```

Criterion also generates an interactive HTML report at `target/criterion/report/index.html`.

## Usage

```rust
use opthash::{ElasticHashMap, ElasticOptions, FunnelHashMap};

let mut map = FunnelHashMap::new();
map.insert("key", 42);
assert_eq!(map.get("key"), Some(&42));

let tuned = ElasticHashMap::<u64, u64>::with_options(ElasticOptions {
    capacity: 1024,
    reserve_fraction: 0.10,
    probe_scale: 12.0,
});
assert_eq!(tuned.len(), 0);
```

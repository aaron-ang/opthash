# opthash

Rust implementations of **Elastic Hashing** and **Funnel Hashing** from [*Optimal Bounds for Open Addressing Without Reordering*](https://arxiv.org/abs/2501.02305) (Farach-Colton, Krapivin, Kuszmaul, 2025).

Both are open-addressing hash maps that achieve optimal expected probe complexity without reordering elements after insertion.

## Data Structures

- **`ElasticHashMap<K, V>`** — Multi-level table with geometrically halving levels. Keys are placed via batch-based insertion across levels using stride-based probing.
- **`FunnelHashMap<K, V>`** — Multi-level bucketed table with a 3/4-ratio geometric progression and a special overflow array (primary + fallback) for keys that don't fit in any level.

Both support `insert`, `get`, `get_mut`, `contains_key`, `remove`, and `clear`. Maps start with zero allocation (`new()`) and grow dynamically on demand.

## Benchmarks

10,000 inserts and 20,000-entry maps for lookups, measured on Apple M1 (aarch64, NEON SIMD):

| Operation  | `std::HashMap` | `ElasticHashMap` | `FunnelHashMap` |
| ---------- | -------------- | ---------------- | --------------- |
| insert     | 50 Melem/s     | 5.4 Melem/s      | 19.8 Melem/s    |
| get (hit)  | 98 Melem/s     | 8.9 Melem/s      | 23.8 Melem/s    |
| get (miss) | 122 Melem/s    | 4.9 Melem/s      | 16.5 Melem/s    |

```
cargo bench --bench throughput
```

## Usage

```rust
use opthash::{ElasticHashMap, FunnelHashMap};

let mut map = FunnelHashMap::new();
map.insert("key", 42);
assert_eq!(map.get("key"), Some(&42));
```

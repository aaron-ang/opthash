# opthash

[![Crates.io](https://img.shields.io/crates/v/opthash?logo=rust&label=crates.io)](https://crates.io/crates/opthash)
[![PyPI](https://img.shields.io/pypi/v/opthash?logo=pypi&logoColor=white&label=pypi)](https://pypi.org/project/opthash/)
[![CI](https://github.com/aaron-ang/opthash/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-ang/opthash/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/opthash?logo=python&logoColor=white)](https://pypi.org/project/opthash/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

Rust implementations of **Elastic Hashing** and **Funnel Hashing** from [_Optimal Bounds for Open Addressing Without Reordering_](https://arxiv.org/abs/2501.02305) (Farach-Colton, Krapivin, Kuszmaul, 2025).

Both are open-addressing hash maps that achieve optimal expected probe complexity without reordering elements after insertion.

## Data Structures

Both maps share a common core: `RawTable`-backed multi-level layouts, 7-bit fingerprint control bytes, SIMD control-byte scans for occupancy + lookup, tombstone accounting, and per-level Lemire fastmod magics for the hash → slot mapping. The default `BuildHasher` is [`foldhash`](https://crates.io/crates/foldhash).

- **`ElasticHashMap<K, V>`** — Flat `RawTable` per level wsith geometrically halving capacities; insertion uses per-level probe budgets + coprime group steps.
- **`FunnelHashMap<K, V>`** — Bucketed levels plus a split special array: `primary` (group-probed) and `fallback` (two-choice buckets).

Both support `insert`, `get`, `get_mut`, `contains_key`, `remove`, and `clear`. Maps start with zero allocation (`new()`) and grow dynamically on demand. Advanced tuning is available through `ElasticOptions`, `FunnelOptions`, and `with_options(...)`.

## Usage

### Rust

```bash
cargo add opthash
```

```rust
use opthash::{ElasticHashMap, ElasticOptions, FunnelHashMap, FunnelOptions};

let mut map = ElasticHashMap::new();
map.insert("key", 42);
assert_eq!(map.get("key"), Some(&42));

let mut map = ElasticHashMap::with_options(ElasticOptions {
    capacity: 1024,
    reserve_fraction: 0.10,
    probe_scale: 12.0,
});
map.insert("key", 42);
assert_eq!(map.get("key"), Some(&42));

let mut map = FunnelHashMap::with_options(FunnelOptions {
    capacity: 1024,
    reserve_fraction: 0.10,
    primary_probe_limit: Some(8),
});
map.insert("key", 42);
assert_eq!(map.get("key"), Some(&42));
```

### Python

```bash
pip install opthash
```

```python
from opthash import ElasticHashMap, ElasticOptions, FunnelHashMap, FunnelOptions

m = ElasticHashMap()
m["key"] = 42
assert m["key"] == 42
assert "key" in m and len(m) == 1

m = ElasticHashMap.with_options(ElasticOptions(
    capacity=1024, reserve_fraction=0.10, probe_scale=12.0,
))

m = FunnelHashMap.with_options(FunnelOptions(
    capacity=1024, reserve_fraction=0.10, primary_probe_limit=8,
))
```

## Layout Sketch

```text
RawTable (shared by both maps)
==============================

  fp = fingerprint (7-bit control byte)
  kv = key-value entry, __ = empty, xx = tombstone

  Single allocation, slots first, controls at the end:

  data_ptr ► [kv][kv][  ][kv][  ][kv]... [pad] [fp][fp][__][xx][__][fp]...
             └──── slots (T-aligned) ────┘     └─ controls (16-aligned) ──┘
                                               ▲ ctrl_offset

  No per-group metadata. Occupancy is derived from SIMD scans of the control bytes
  (eq_mask_16 for fingerprints, free_mask_16 for free slots).


ElasticHashMap
==============

  levels: Vec<Level>

    Level 0    RawTable  (largest, ~half of total capacity)
    Level 1    RawTable  (geometrically halved)
    Level 2    ...

    per-level  len, tombstones, half_reserve_slot_threshold,
               limited_probe_budgets, group_steps, salt,
               group_count_magic, step_count_magic

  table-wide   len, capacity, max_insertions, reserve_fraction,
               probe_scale, batch_plan, current_batch_index,
               batch_remaining, max_populated_level, hash_builder


FunnelHashMap
=============

  levels: Vec<BucketLevel>

    Level 0
      slots:     kv kv __ __ ... kv kv __ __ ... kv ...
      controls:  fp fp __ __ ... fp fp __ __ ... fp ...
                 └── bucket 0 ──┘└── bucket 1 ──┘

    Level 1    (same layout, smaller buckets)
    ...

    per-level  len, tombstones, bucket_size, bucket_count,
               salt, bucket_count_magic

  special: SpecialArray

    primary    RawTable, group-probed (like elastic)
    (paper B)  len, group_summaries, group_tombstones, group_steps

    fallback   RawTable, two-choice bucketed
    (paper C)  len, tombstones, bucket_size, bucket_count

  table-wide   len, capacity, max_insertions, reserve_fraction,
               primary_probe_limit, max_populated_level, hash_builder
```

## Benchmarks

See [benches/README.md](benches/README.md) for bench target layout, charts, CLI flags, chart regeneration, and flamegraph profiling.

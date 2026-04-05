# opthash

Rust implementations of **Elastic Hashing** and **Funnel Hashing** from [*Optimal Bounds for Open Addressing Without Reordering*](https://arxiv.org/abs/2501.02305) (Farach-Colton, Krapivin, Kuszmaul, 2025).

Both are open-addressing hash maps that achieve optimal expected probe complexity without reordering elements after insertion.

## Data Structures

* **`ElasticHashMap<K, V>`** — Multi-level table with geometrically halving levels. Each level is a `RawTable` plus probe budgets, group steps, and tombstone accounting.
* **`FunnelHashMap<K, V>`** — Multi-level bucketed table with per-bucket metadata and a split special array: `primary` (group-probed) plus `fallback` (two-choice buckets).

Both support `insert`, `get`, `get_mut`, `contains_key`, `remove`, and `clear`. Maps start with zero allocation (`new()`) and grow dynamically on demand. Advanced tuning is available through `ElasticOptions`, `FunnelOptions`, and `with_options(...)`.

## Usage

```rust
use opthash::{ElasticHashMap, ElasticOptions, FunnelHashMap, FunnelOptions};

let mut map = FunnelHashMap::new();
map.insert("key", 42);
assert_eq!(map.get("key"), Some(&42));

let tuned_funnel = FunnelHashMap::<u64, u64>::with_options(FunnelOptions {
    capacity: 1024,
    reserve_fraction: 0.10,
    primary_probe_limit: Some(8),
});
assert_eq!(tuned_funnel.len(), 0);

let tuned = ElasticHashMap::<u64, u64>::with_options(ElasticOptions {
    capacity: 1024,
    reserve_fraction: 0.10,
    probe_scale: 12.0,
});
assert_eq!(tuned.len(), 0);
```

### Layout Sketch

```text
RawTable (shared by both maps)
==============================

  Single allocation per table, slots first:

    data_ptr ─► [kv][kv][  ][kv][  ][kv]... [pad] [fp][fp][__][xx][__][fp]...
                 └─── slots (T-aligned) ───┘        └── controls (16-aligned) ──┘
                                                         ▲ ctrl_offset

  fp = fingerprint (7-bit control byte), kv = key-value entry
  __ = empty slot, xx = tombstone

  No per-group metadata — occupancy is derived from SIMD scans of the
  control bytes (eq_mask_16 for fingerprints, free_mask_16 for free slots).


ElasticHashMap
==============

  levels: Vec<Level>

    Level 0   RawTable (largest, ~half of total capacity)
    Level 1   RawTable (geometrically halved)
    Level 2   ...

    per-level   len, tombstones, half_reserve_slot_threshold
                limited_probe_budgets, group_steps, salt

    table-wide  len, capacity, max_insertions, reserve_fraction, probe_scale
                batch_plan, current_batch_index, batch_remaining
                max_populated_level, hash_builder


FunnelHashMap
=============

  levels: Vec<BucketLevel>

    Level 0
      RawTable   [kv kv __ __ | kv __ __ __]... [fp fp __ __ | fp __ __ __]...
                   └─ bucket 0 ─┘ └─ bucket 1 ─┘
      bucket_meta  BucketMeta { summary, live_mask, search_len, live, tombstones }

    Level 1  (same layout, smaller buckets)
    ...

  special: SpecialArray

    primary (paper B)
      RawTable     group-probed like elastic
      per-primary  len, group_summaries, group_tombstones, group_steps

    fallback (paper C)
      RawTable     two-choice bucketed
      per-fallback len, bucket_size, bucket_count
                   bucket_summaries, bucket_live, bucket_tombstones

    table-wide  len, capacity, max_insertions, reserve_fraction
                primary_probe_limit, max_populated_level, hash_builder
```

## Benchmarks

All benchmarks on Apple M1 (aarch64, NEON SIMD) via Criterion. `std::HashMap` uses the same `RandomState` (SipHash) hasher as both custom maps.

### Throughput

![Throughput speedup chart](assets/benchmark-speedup.svg)

### Latency

![Latency chart](assets/benchmark-latency.svg)

### Running benchmarks

```bash
cargo bench --bench benchmarks            # run all throughput + latency benchmarks
uv run scripts/generate_speedup_chart.py  # regenerate charts from Criterion JSON
```

Criterion also generates an interactive HTML report at `target/criterion/report/index.html`.

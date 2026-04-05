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
ElasticHashMap
==============
RawTable stride = META_STRIDE (32)

levels: Vec<Level>

  fp = fingerprint (7-bit control byte), kv = key-value entry,
  __ = empty slot, xx = tombstone, L/T/F = live count / tombstone count / full flag

  Each group is a 32-byte block: 16 control bytes + 3 metadata bytes + padding.
  Controls and GroupMeta share one cache line, halving misses per probe.

  Level 0
    data_ptr ─┬─ group 0 ─────────────────────────────────────────┐
              │  [fp][fp][__][xx][__][fp][__][__]...[__][fp] [L][T][F] ...pad
              │   └── 16 control bytes (SIMD scan) ──┘       │   │  │
              │                                         live ┘   │  └ full
              ├─ group 1                            tombstones ──┘
              │  [fp][__][__][__][fp][__][fp][__]...[fp][__] [L][T][F] ...pad
              └─ ...
    slots_ptr   [kv][kv][  ][kv][  ][kv][  ][  ] ...

    per-level   len, tombstones, half_reserve_slot_threshold
                limited_probe_budgets, group_steps, salt

  Level 1  (same block layout, geometrically halved capacity)
  Level 2  ...

  table-wide
    len, capacity, max_insertions, reserve_fraction, probe_scale
    batch_plan, current_batch_index, batch_remaining
    max_populated_level, hash_builder


FunnelHashMap
=============

levels: Vec<BucketLevel>
  RawTable stride = COMPACT_STRIDE (16) — flat control bytes, no embedded meta.

  Level 0
    data_ptr    [fp][fp][__][__] [fp][__][__][__] ...
                 └── bucket 0 ──┘ └── bucket 1 ──┘
    slots_ptr   [kv][kv][  ][  ] [kv][  ][  ][  ] ...
    bucket_meta BucketMeta { summary, live_mask, search_len, live, tombstones }

  Level 1  (same layout, smaller buckets)
  ...

special: SpecialArray

  primary (paper B)
    RawTable stride = META_STRIDE (32) — same block layout as elastic.
    data_ptr ── group 0: [fp][__][fp][__]...[__][fp] [L][T][F] ...pad
    slots_ptr   [kv][  ][kv][  ] ...
    per-primary len, group_summaries, group_tombstones, group_steps

  fallback (paper C)
    RawTable stride = COMPACT_STRIDE (16) — flat control bytes.
    data_ptr    [fp][__] [__][fp] ...
                 └─ b0 ─┘ └─ b1 ─┘
    slots_ptr   [kv][  ] [  ][kv] ...
    per-fallback len, bucket_size, bucket_count
                 bucket_summaries, bucket_live, bucket_tombstones

  table-wide
    len, capacity, max_insertions, reserve_fraction
    primary_probe_limit, max_populated_level, hash_builder
```

## Benchmarks

All benchmarks on Apple M1 (aarch64, NEON SIMD) via Criterion. `std::HashMap` uses the same `RandomState` (SipHash) hasher as both custom maps.

### Throughput

Speedup relative to `std::HashMap` (1.0x baseline). Higher is better:

![Throughput speedup chart](assets/benchmark-speedup.svg)

### Latency

Per-lookup latency for a single `get()` call at different map sizes (100 to 10M entries). As the working set exceeds L2 cache, elastic hashing's random probing converges toward hashbrown's linear probing — at 10M entries elastic is only 1.15x slower:

![Latency chart](assets/benchmark-latency.svg)

### Running benchmarks

```bash
cargo bench --bench benchmarks          # run all throughput + latency benchmarks
uv run scripts/generate_speedup_chart.py  # regenerate charts from Criterion JSON
```

Criterion also generates an interactive HTML report at `target/criterion/report/index.html`.

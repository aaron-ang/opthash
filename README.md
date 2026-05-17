# opthash

[![Crates.io](https://img.shields.io/crates/v/opthash?logo=rust&label=crates.io)](https://crates.io/crates/opthash)
[![PyPI](https://img.shields.io/pypi/v/opthash?logo=pypi&logoColor=white&label=pypi)](https://pypi.org/project/opthash/)
[![CI](https://github.com/aaron-ang/opthash-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-ang/opthash-rs/actions/workflows/ci.yml)
[![Release](https://github.com/aaron-ang/opthash-rs/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/aaron-ang/opthash-rs/actions/workflows/release.yml)
[![Python](https://img.shields.io/pypi/pyversions/opthash?logo=python&logoColor=white)](https://pypi.org/project/opthash/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Rust implementations of **Elastic Hashing** and **Funnel Hashing** from _Optimal Bounds for Open Addressing Without Reordering_ (Farach-Colton, Krapivin, Kuszmaul, 2025) — see [References](#references) [^fkk2024].

Both are open-addressing hash maps that achieve optimal expected probe complexity without reordering elements after insertion.

## Data Structures

Both maps share a common core: `RawTable`-backed multi-level layouts, 7-bit fingerprint control bytes, SIMD control-byte scans for occupancy + lookup, tombstone accounting, and SwissTable-style triangular probing within every level [^swisstable] [^cppcon2017] [^hashbrown]. Per-level `group_count` is rounded up to a power of two so `(idx + delta) & mask` wraps in one op. The default `BuildHasher` is [`foldhash`](https://crates.io/crates/foldhash) [^foldhash].

- **`ElasticHashMap<K, V>`** — Flat `RawTable` per level with geometrically halving capacities; insertion uses per-level probe budgets.
- **`FunnelHashMap<K, V>`** — Bucketed levels plus a split special array: `primary` (group-probed) and `fallback` (two-choice buckets).

Memory note: pow2 rounding can inflate slot count by up to ~2× the requested capacity; the logical capacity (used for `max_insertions` / resize) is unchanged.

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
from opthash import ElasticHashMap, FunnelHashMap

m = ElasticHashMap()
m["key"] = 42
assert m["key"] == 42
assert "key" in m and len(m) == 1

m = ElasticHashMap.with_options(
    capacity=1024, reserve_fraction=0.10, probe_scale=12.0
)

m = FunnelHashMap.with_options(
    capacity=1024, reserve_fraction=0.10, primary_probe_limit=8
)
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
               limited_probe_budgets, salt, group_count_mask

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
    (paper B)  len, group_count_mask, group_summaries, group_tombstones

    fallback   RawTable, two-choice bucketed
    (paper C)  len, tombstones, bucket_size, bucket_count

  table-wide   len, capacity, max_insertions, reserve_fraction,
               primary_probe_limit, max_populated_level, hash_builder
```

## Benchmarks

See [benches/README.md](benches/README.md) for bench target layout, charts, CLI flags, chart regeneration, and flamegraph profiling.

## References

[^fkk2024]: Martín Farach-Colton, Andrew Krapivin, William Kuszmaul. *Optimal Bounds for Open Addressing Without Reordering* (2025). arXiv: <https://arxiv.org/abs/2501.02305>. Establishes the elastic and funnel hashing schemes implemented in [`src/elastic.rs`](src/elastic.rs) and [`src/funnel.rs`](src/funnel.rs); the funnel "special array" split into `primary` (group-probed, paper B) and `fallback` (two-choice, paper C) follows the paper's construction directly.

[^cw1979]: J. Lawrence Carter, Mark N. Wegman. *Universal Classes of Hash Functions* (STOC 1977 / JCSS 1979). DOI: <https://doi.org/10.1016/0022-0000(79)90044-8>. Foundational hash-based probing model the FKK bounds rely on; the per-level `salt` re-randomization in `Level`/`BucketLevel` (see `level_salt` in [`src/common/math.rs`](src/common/math.rs)) follows the universal-hashing assumption.

[^swisstable]: Abseil. *SwissTable design notes*. <https://abseil.io/about/design/swisstables>. Source of the 7-bit fingerprint control-byte layout + SIMD group scans used by `RawTable` (see [`src/common/control.rs`](src/common/control.rs), [`src/common/simd.rs`](src/common/simd.rs)) and the triangular `(idx + delta) & mask` probe sequence used in `ElasticHashMap::triangular_group_start` and `FunnelHashMap::special_primary_triangular_start`.

[^cppcon2017]: Matt Kulukundis. *Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step* (CppCon 2017). <https://www.youtube.com/watch?v=ncHmEUmJZf4>. Talk introducing the SwissTable design referenced above.

[^hashbrown]: `hashbrown` — Rust port of SwissTable. <https://github.com/rust-lang/hashbrown>. Used as the absolute throughput ceiling in the Criterion benches (see [benches/README.md](benches/README.md)).

[^foldhash]: `foldhash` crate. <https://crates.io/crates/foldhash>. Default `BuildHasher` (`foldhash::fast::RandomState`) wired up in [`src/common/mod.rs`](src/common/mod.rs).

[^prefetch2007]: Shimin Chen, Anastassia Ailamaki, Phillip B. Gibbons, Todd C. Mowry. *Improving Hash Join Performance through Prefetching* (ACM TODS 2007). PDF: <https://www.cs.cmu.edu/~chensm/papers/hashjoin_tods_preliminary.pdf>. Motivates the intra-probe `prefetch_read(group_data_ptr(next))` issued one group ahead in `find_in_special_primary` / `find_in_special_primary_with_candidate` (see [`src/funnel.rs`](src/funnel.rs)).

[^fastmod]: Daniel Lemire. *Faster Remainders when the Divisor is a Constant: Beating Compilers and libdivide* (2019). <https://lemire.me/blog/2019/02/08/faster-remainders-when-the-divisor-is-a-constant-beating-compilers-and-libdivide/>. Algorithm behind `fastmod_magic` / `fastmod_u32` in [`src/common/math.rs`](src/common/math.rs), used by `BucketLevel::bucket_index` to map a hash to a bucket without a hardware divide.

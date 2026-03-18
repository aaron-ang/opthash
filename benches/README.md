# Benchmarking

This project uses Criterion to compare:

- `std::collections::HashMap`
- `opthash::ElasticHashMap`
- `opthash::FunnelHashMap`

## Throughput workloads

`benches/throughput.rs` measures:

1. `insert_throughput`
2. `get_hit_throughput`
3. `get_miss_throughput`
4. `tiny_lookup_throughput`
5. `delete_heavy_throughput`
6. `resize_heavy_throughput`
7. `mixed_lookup_throughput`

The small-map workload is meant to exercise the internal tiny-table engine. The delete-heavy and resize-heavy workloads are there to expose tombstone handling and growth costs instead of only steady-state inserts.

## Internal-path workloads

`benches/internal_paths.rs` measures narrower hot paths:

1. `internal_control_group_scan`
2. `internal_elastic_miss_path`
3. `internal_funnel_miss_path`
4. `internal_resize_cost`

`internal_control_group_scan` focuses on the packed control-byte scanning layer directly. The two miss-path benchmarks isolate lookup rejection behavior in the elastic and funnel layouts. `internal_resize_cost` measures repeated growth from a deliberately small initial capacity.

## Run

```bash
cargo bench --bench throughput
cargo bench --bench internal_paths
```

## Visual reports

Criterion writes HTML reports under:

- `target/criterion/report/index.html`

Per-workload pages are nested below their benchmark group names, for example:

- `target/criterion/insert_throughput/report/index.html`
- `target/criterion/tiny_lookup_throughput/report/index.html`
- `target/criterion/internal_control_group_scan/report/index.html`

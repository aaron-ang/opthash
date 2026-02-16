# Benchmarking

This project includes a Criterion benchmark that compares:

- `std::collections::HashMap`
- `opthash::ElasticHashMap`
- `opthash::FunnelHashMap`

## Workloads

The benchmark suite (`benches/throughput.rs`) measures throughput for:

1. Insert throughput (`insert_throughput`)
2. Successful lookup throughput (`get_hit_throughput`)
3. Unsuccessful lookup throughput (`get_miss_throughput`)

## Run

```bash
cargo bench --bench throughput
```

## Visual report

Criterion generates an HTML report with plots:

- `target/criterion/report/index.html`

Per-workload pages are also available under:

- `target/criterion/insert_throughput/report/index.html`
- `target/criterion/get_hit_throughput/report/index.html`
- `target/criterion/get_miss_throughput/report/index.html`

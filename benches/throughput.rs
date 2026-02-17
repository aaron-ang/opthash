use std::collections::HashMap as StdHashMap;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use opthash::{ElasticHashMap, FunnelHashMap};

const INSERT_COUNT: usize = 10_000;
const LOOKUP_MAP_SIZE: usize = 20_000;
const HIT_LOOKUP_COUNT: usize = 200_000;
const MISS_LOOKUP_COUNT: usize = 20_000;

fn key_at(index: usize) -> u64 {
    // Large odd multiplier gives a deterministic permutation-like key stream.
    (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn make_pairs(count: usize) -> Vec<(u64, u64)> {
    (0..count)
        .map(|idx| {
            let key = key_at(idx);
            (key, key ^ 0xA5A5_A5A5_A5A5_A5A5)
        })
        .collect()
}

fn build_std_map(pairs: &[(u64, u64)]) -> StdHashMap<u64, u64> {
    let mut map = StdHashMap::with_capacity(pairs.len() * 2);
    for &(key, value) in pairs {
        map.insert(key, value);
    }
    map
}

fn build_elastic_map(pairs: &[(u64, u64)]) -> ElasticHashMap<u64, u64> {
    let mut map = ElasticHashMap::with_capacity(pairs.len() * 2);
    for &(key, value) in pairs {
        map.insert(key, value);
    }
    map
}

fn build_funnel_map(pairs: &[(u64, u64)]) -> FunnelHashMap<u64, u64> {
    let mut map = FunnelHashMap::with_capacity(pairs.len() * 2);
    for &(key, value) in pairs {
        map.insert(key, value);
    }
    map
}

fn bench_insert_throughput(c: &mut Criterion) {
    let pairs = make_pairs(INSERT_COUNT);
    let mut group = c.benchmark_group("insert_throughput");
    group.throughput(Throughput::Elements(INSERT_COUNT as u64));

    group.bench_function("std", |b| {
        b.iter_batched_ref(
            || StdHashMap::with_capacity(INSERT_COUNT * 2),
            |map| {
                for &(key, value) in &pairs {
                    map.insert(black_box(key), black_box(value));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("elastic", |b| {
        b.iter_batched_ref(
            || ElasticHashMap::with_capacity(INSERT_COUNT * 2),
            |map| {
                for &(key, value) in &pairs {
                    map.insert(black_box(key), black_box(value));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("funnel", |b| {
        b.iter_batched_ref(
            || FunnelHashMap::with_capacity(INSERT_COUNT * 2),
            |map| {
                for &(key, value) in &pairs {
                    map.insert(black_box(key), black_box(value));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_get_hit_throughput(c: &mut Criterion) {
    let pairs = make_pairs(LOOKUP_MAP_SIZE);
    let query_keys: Vec<u64> = (0..HIT_LOOKUP_COUNT)
        .map(|idx| pairs[idx % LOOKUP_MAP_SIZE].0)
        .collect();

    let mut group = c.benchmark_group("get_hit_throughput");
    group.throughput(Throughput::Elements(HIT_LOOKUP_COUNT as u64));

    group.bench_function("std", |b| {
        b.iter_batched(
            || build_std_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("elastic", |b| {
        b.iter_batched(
            || build_elastic_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("funnel", |b| {
        b.iter_batched(
            || build_funnel_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_get_miss_throughput(c: &mut Criterion) {
    let pairs = make_pairs(LOOKUP_MAP_SIZE);
    let query_keys: Vec<u64> = (0..MISS_LOOKUP_COUNT)
        .map(|idx| key_at(idx + LOOKUP_MAP_SIZE + 10_000_000))
        .collect();

    let mut group = c.benchmark_group("get_miss_throughput");
    group.throughput(Throughput::Elements(MISS_LOOKUP_COUNT as u64));

    group.bench_function("std", |b| {
        b.iter_batched(
            || build_std_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("elastic", |b| {
        b.iter_batched(
            || build_elastic_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("funnel", |b| {
        b.iter_batched(
            || build_funnel_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_insert_throughput, bench_get_hit_throughput, bench_get_miss_throughput
);
criterion_main!(benches);

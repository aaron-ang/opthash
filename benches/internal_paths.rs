use std::collections::HashMap as StdHashMap;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use opthash::{
    ElasticHashMap, FunnelHashMap,
    bench_support::{CTRL_EMPTY, control_match_fingerprint_group, control_match_free_group},
};

const CONTROL_SCAN_COUNT: usize = 1_024;
const CONTROL_GROUP_OPS: usize = 4_096;
const MAP_SIZE: usize = 20_000;
const MISS_LOOKUP_COUNT: usize = 100_000;
const RESIZE_TRIGGER_COUNT: usize = 2_000;

fn key_at(index: usize) -> u64 {
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

fn bench_control_group_scan(c: &mut Criterion) {
    let mut controls = vec![7u8; CONTROL_SCAN_COUNT];
    for idx in (0..CONTROL_SCAN_COUNT).step_by(31) {
        controls[idx] = 9;
    }
    for idx in (15..CONTROL_SCAN_COUNT).step_by(53) {
        controls[idx] = CTRL_EMPTY;
    }

    let mut group = c.benchmark_group("internal_control_group_scan");
    group.throughput(Throughput::Elements(CONTROL_GROUP_OPS as u64));

    group.bench_function("find_fingerprint", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for chunk in controls.chunks_exact(32).take(CONTROL_GROUP_OPS / 32) {
                total = total.wrapping_add(black_box(
                    control_match_fingerprint_group(chunk, 9).count_ones() as usize,
                ));
            }
            black_box(total)
        });
    });

    group.bench_function("find_free", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for chunk in controls.chunks_exact(32).take(CONTROL_GROUP_OPS / 32) {
                total = total.wrapping_add(black_box(
                    control_match_free_group(chunk).count_ones() as usize
                ));
            }
            black_box(total)
        });
    });

    group.finish();
}

fn bench_elastic_miss_path(c: &mut Criterion) {
    let pairs = make_pairs(MAP_SIZE);
    let miss_keys: Vec<u64> = (0..MISS_LOOKUP_COUNT)
        .map(|idx| key_at(idx + 50_000_000))
        .collect();

    let mut group = c.benchmark_group("internal_elastic_miss_path");
    group.throughput(Throughput::Elements(MISS_LOOKUP_COUNT as u64));
    group.bench_function("elastic", |b| {
        b.iter_batched(
            || {
                let mut map = ElasticHashMap::with_capacity(MAP_SIZE * 2);
                for &(key, value) in &pairs {
                    map.insert(key, value);
                }
                map
            },
            |map| {
                for key in &miss_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_funnel_miss_path(c: &mut Criterion) {
    let pairs = make_pairs(MAP_SIZE);
    let miss_keys: Vec<u64> = (0..MISS_LOOKUP_COUNT)
        .map(|idx| key_at(idx + 80_000_000))
        .collect();

    let mut group = c.benchmark_group("internal_funnel_miss_path");
    group.throughput(Throughput::Elements(MISS_LOOKUP_COUNT as u64));
    group.bench_function("funnel", |b| {
        b.iter_batched(
            || {
                let mut map = FunnelHashMap::with_capacity(MAP_SIZE * 2);
                for &(key, value) in &pairs {
                    map.insert(key, value);
                }
                map
            },
            |map| {
                for key in &miss_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_resize_cost(c: &mut Criterion) {
    let pairs = make_pairs(RESIZE_TRIGGER_COUNT);
    let mut group = c.benchmark_group("internal_resize_cost");
    group.throughput(Throughput::Elements(RESIZE_TRIGGER_COUNT as u64));

    group.bench_function("std", |b| {
        b.iter(|| {
            let mut map = StdHashMap::with_capacity(8);
            for &(key, value) in &pairs {
                black_box(map.insert(black_box(key), black_box(value)));
            }
            black_box(map.len())
        });
    });

    group.bench_function("elastic", |b| {
        b.iter(|| {
            let mut map = ElasticHashMap::with_capacity(8);
            for &(key, value) in &pairs {
                black_box(map.insert(black_box(key), black_box(value)));
            }
            black_box(map.len())
        });
    });

    group.bench_function("funnel", |b| {
        b.iter(|| {
            let mut map = FunnelHashMap::with_capacity(8);
            for &(key, value) in &pairs {
                black_box(map.insert(black_box(key), black_box(value)));
            }
            black_box(map.len())
        });
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        bench_control_group_scan,
        bench_elastic_miss_path,
        bench_funnel_miss_path,
        bench_resize_cost
);
criterion_main!(benches);

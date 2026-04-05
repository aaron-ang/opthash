use std::collections::HashMap as StdHashMap;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use opthash::{ElasticHashMap, FunnelHashMap};

const INSERT_COUNT: usize = 10_000;
const LOOKUP_MAP_SIZE: usize = 20_000;
const HIT_LOOKUP_COUNT: usize = 200_000;
const MISS_LOOKUP_COUNT: usize = 20_000;
const TINY_MAP_SIZE: usize = 32;
const TINY_LOOKUP_COUNT: usize = 20_000;
const DELETE_MAP_SIZE: usize = 12_000;
const DELETE_OP_COUNT: usize = 6_000;
const RESIZE_INSERT_COUNT: usize = 8_000;
const MIXED_LOOKUP_COUNT: usize = 100_000;

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

fn bench_tiny_lookup_throughput(c: &mut Criterion) {
    let pairs = make_pairs(TINY_MAP_SIZE);
    let query_keys: Vec<u64> = (0..TINY_LOOKUP_COUNT)
        .map(|idx| {
            if idx % 2 == 0 {
                pairs[idx % TINY_MAP_SIZE].0
            } else {
                key_at(idx + 5_000_000)
            }
        })
        .collect();

    let mut group = c.benchmark_group("tiny_lookup_throughput");
    group.throughput(Throughput::Elements(TINY_LOOKUP_COUNT as u64));

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
            || {
                let mut map = ElasticHashMap::with_capacity(TINY_MAP_SIZE);
                for &(key, value) in &pairs {
                    map.insert(key, value);
                }
                map
            },
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
            || {
                let mut map = FunnelHashMap::with_capacity(TINY_MAP_SIZE);
                for &(key, value) in &pairs {
                    map.insert(key, value);
                }
                map
            },
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

fn bench_delete_heavy_throughput(c: &mut Criterion) {
    let initial_pairs = make_pairs(DELETE_MAP_SIZE);
    let replacement_pairs: Vec<(u64, u64)> = (0..DELETE_OP_COUNT)
        .map(|idx| {
            let key = key_at(idx + 20_000_000);
            (key, key ^ 0x5A5A_5A5A_5A5A_5A5A)
        })
        .collect();

    let mut group = c.benchmark_group("delete_heavy_throughput");
    group.throughput(Throughput::Elements((DELETE_OP_COUNT * 2) as u64));

    group.bench_function("std", |b| {
        b.iter_batched(
            || build_std_map(&initial_pairs),
            |mut map| {
                for idx in 0..DELETE_OP_COUNT {
                    black_box(map.remove(black_box(&initial_pairs[idx].0)));
                    let (key, value) = replacement_pairs[idx];
                    black_box(map.insert(black_box(key), black_box(value)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("elastic", |b| {
        b.iter_batched(
            || build_elastic_map(&initial_pairs),
            |mut map| {
                for idx in 0..DELETE_OP_COUNT {
                    black_box(map.remove(black_box(&initial_pairs[idx].0)));
                    let (key, value) = replacement_pairs[idx];
                    black_box(map.insert(black_box(key), black_box(value)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("funnel", |b| {
        b.iter_batched(
            || build_funnel_map(&initial_pairs),
            |mut map| {
                for idx in 0..DELETE_OP_COUNT {
                    black_box(map.remove(black_box(&initial_pairs[idx].0)));
                    let (key, value) = replacement_pairs[idx];
                    black_box(map.insert(black_box(key), black_box(value)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_resize_heavy_throughput(c: &mut Criterion) {
    let pairs = make_pairs(RESIZE_INSERT_COUNT);
    let mut group = c.benchmark_group("resize_heavy_throughput");
    group.throughput(Throughput::Elements(RESIZE_INSERT_COUNT as u64));

    group.bench_function("std", |b| {
        b.iter(|| {
            let mut map = StdHashMap::new();
            for &(key, value) in &pairs {
                black_box(map.insert(black_box(key), black_box(value)));
            }
            black_box(map.len())
        });
    });

    group.bench_function("elastic", |b| {
        b.iter(|| {
            let mut map = ElasticHashMap::new();
            for &(key, value) in &pairs {
                black_box(map.insert(black_box(key), black_box(value)));
            }
            black_box(map.len())
        });
    });

    group.bench_function("funnel", |b| {
        b.iter(|| {
            let mut map = FunnelHashMap::new();
            for &(key, value) in &pairs {
                black_box(map.insert(black_box(key), black_box(value)));
            }
            black_box(map.len())
        });
    });

    group.finish();
}

fn bench_mixed_lookup_throughput(c: &mut Criterion) {
    let pairs = make_pairs(LOOKUP_MAP_SIZE);
    let query_keys: Vec<u64> = (0..MIXED_LOOKUP_COUNT)
        .map(|idx| {
            if idx % 3 == 0 {
                key_at(idx + 50_000_000)
            } else {
                pairs[idx % LOOKUP_MAP_SIZE].0
            }
        })
        .collect();

    let mut group = c.benchmark_group("mixed_lookup_throughput");
    group.throughput(Throughput::Elements(MIXED_LOOKUP_COUNT as u64));

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

const LATENCY_SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];

fn bench_get_hit_latency(c: &mut Criterion) {
    for &size in LATENCY_SIZES {
        let pairs = make_pairs(size);
        let query_keys: Vec<u64> = (0..size).map(|idx| pairs[idx].0).collect();

        let label = if size >= 1_000_000 {
            format!("{}M", size / 1_000_000)
        } else if size >= 1_000 {
            format!("{}K", size / 1_000)
        } else {
            format!("{size}")
        };

        let mut group = c.benchmark_group(format!("get_hit_latency_{label}"));

        group.bench_function("std", |b| {
            let map = build_std_map(&pairs);
            let mut i = 0;
            b.iter(|| {
                let key = &query_keys[i % size];
                i = i.wrapping_add(1);
                black_box(map.get(black_box(key)))
            });
        });

        group.bench_function("elastic", |b| {
            let map = build_elastic_map(&pairs);
            let mut i = 0;
            b.iter(|| {
                let key = &query_keys[i % size];
                i = i.wrapping_add(1);
                black_box(map.get(black_box(key)))
            });
        });

        group.bench_function("funnel", |b| {
            let map = build_funnel_map(&pairs);
            let mut i = 0;
            b.iter(|| {
                let key = &query_keys[i % size];
                i = i.wrapping_add(1);
                black_box(map.get(black_box(key)))
            });
        });

        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        bench_insert_throughput,
        bench_get_hit_throughput,
        bench_get_miss_throughput,
        bench_tiny_lookup_throughput,
        bench_delete_heavy_throughput,
        bench_resize_heavy_throughput,
        bench_mixed_lookup_throughput,
        bench_get_hit_latency
);
criterion_main!(benches);

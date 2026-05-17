mod common;

use std::collections::HashMap as StdHashMap;
use std::hint::black_box;

use std::path::Path;
use std::time::Duration;

use common::{
    LATENCY_SIZES, VALUE_XOR_MIX_ALT, build_elastic_map, build_funnel_map, build_hashbrown_map,
    build_std_map, key_at, make_pairs, size_label,
};
use criterion::{
    BatchSize, Criterion, Throughput, criterion_group, criterion_main, profiler::Profiler,
};
use hashbrown::HashMap as HashbrownMap;
use opthash::{ElasticHashMap, FunnelHashMap};
use pprof::{ProfilerGuard, flamegraph::Options as FlamegraphOptions};

struct FlamegraphProfiler {
    frequency: i32,
    active: Option<ProfilerGuard<'static>>,
}

impl FlamegraphProfiler {
    fn new() -> Self {
        Self {
            frequency: 997,
            active: None,
        }
    }
}

impl Profiler for FlamegraphProfiler {
    fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
        self.active = Some(ProfilerGuard::new(self.frequency).unwrap());
    }

    fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
        if let Some(guard) = self.active.take() {
            let report = guard.report().build().unwrap();
            let mut opts = FlamegraphOptions::default();
            opts.deterministic = true;
            std::fs::create_dir_all(benchmark_dir).unwrap();
            let path = benchmark_dir.join("flamegraph.svg");
            let file = std::fs::File::create(&path).unwrap();
            report.flamegraph_with_options(file, &mut opts).unwrap();
        }
    }
}

const INSERT_COUNT: usize = 10_000;
const LOOKUP_MAP_SIZE: usize = 20_000;
const HIT_LOOKUP_COUNT: usize = 200_000;
const MISS_LOOKUP_COUNT: usize = 20_000;
const TINY_MAP_SIZE: usize = 32;
const TINY_LOOKUP_COUNT: usize = 20_000;
const DELETE_MAP_SIZE: usize = 12_000;
const DELETE_OP_COUNT: usize = 6_000;

/// `2 * N → with_capacity(32768)` puts elastic level 0 on pow2 group_count,
/// triggering the triangular probe path. Pinned by
/// `bench_pow2_capacity_triggers_triangular`.
const LOOKUP_MAP_SIZE_POW2: usize = 16_384;
/// `2 * N → with_capacity(27104)` puts funnel SpecialPrimary on pow2
/// group_count. Pinned by `bench_pow2_capacity_triggers_special_primary_triangular`.
const FUNNEL_POW2_MAP_SIZE: usize = 13_552;
const RESIZE_INSERT_COUNT: usize = 8_000;
const MIXED_LOOKUP_COUNT: usize = 100_000;

// `multi_get` batched lookup: 1M-entry map (spills L3), 1000-key bursts.
const GET_MANY_MAP_SIZE: usize = 1_000_000;
const GET_MANY_BATCH: usize = 1_000;
const GET_MANY_TOTAL: usize = 100_000;

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

    group.bench_function("hashbrown", |b| {
        b.iter_batched_ref(
            || HashbrownMap::<u64, u64>::with_capacity(INSERT_COUNT * 2),
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
            BatchSize::LargeInput,
        );
    });

    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || build_hashbrown_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
        );
    });

    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || build_hashbrown_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
        );
    });

    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || {
                let mut map = HashbrownMap::with_capacity(TINY_MAP_SIZE);
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
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_delete_heavy_throughput(c: &mut Criterion) {
    let initial_pairs = make_pairs(DELETE_MAP_SIZE);
    let replacement_pairs: Vec<(u64, u64)> = (0..DELETE_OP_COUNT)
        .map(|idx| {
            let key = key_at(idx + 20_000_000);
            (key, key ^ VALUE_XOR_MIX_ALT)
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

    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || build_hashbrown_map(&initial_pairs),
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
        b.iter_batched(
            StdHashMap::new,
            |mut map| {
                for &(key, value) in &pairs {
                    black_box(map.insert(black_box(key), black_box(value)));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            HashbrownMap::<u64, u64>::new,
            |mut map| {
                for &(key, value) in &pairs {
                    black_box(map.insert(black_box(key), black_box(value)));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("elastic", |b| {
        b.iter_batched(
            ElasticHashMap::new,
            |mut map| {
                for &(key, value) in &pairs {
                    black_box(map.insert(black_box(key), black_box(value)));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("funnel", |b| {
        b.iter_batched(
            FunnelHashMap::new,
            |mut map| {
                for &(key, value) in &pairs {
                    black_box(map.insert(black_box(key), black_box(value)));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
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
            BatchSize::LargeInput,
        );
    });

    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || build_hashbrown_map(&pairs),
            |map| {
                for key in &query_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
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
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_multi_get_batch(c: &mut Criterion) {
    // Naive `.get()` loop vs. pipelined `multi_get_into`. Both arms share
    // the chunk slices and a caller-owned scratch buffer so the only delta
    // is the prefetch. The `*_multi_get` arms keep the per-call alloc cost
    // of the owning wrapper visible.
    let pairs = make_pairs(GET_MANY_MAP_SIZE);
    let query_keys: Vec<u64> = (0..GET_MANY_TOTAL)
        .map(|idx| pairs[idx % GET_MANY_MAP_SIZE].0)
        .collect();
    let chunks: Vec<Vec<&u64>> = query_keys
        .chunks(GET_MANY_BATCH)
        .map(|chunk| chunk.iter().collect())
        .collect();

    let mut group = c.benchmark_group("multi_get_batch");
    group.throughput(Throughput::Elements(GET_MANY_TOTAL as u64));

    group.bench_function("elastic_naive", |b| {
        let map = build_elastic_map(&pairs);
        let mut scratch: Vec<Option<&u64>> = Vec::with_capacity(GET_MANY_BATCH);
        b.iter(|| {
            for chunk in &chunks {
                scratch.clear();
                for key in chunk {
                    scratch.push(map.get(black_box(*key)));
                }
                black_box(&scratch);
            }
        });
    });

    group.bench_function("funnel_naive", |b| {
        let map = build_funnel_map(&pairs);
        let mut scratch: Vec<Option<&u64>> = Vec::with_capacity(GET_MANY_BATCH);
        b.iter(|| {
            for chunk in &chunks {
                scratch.clear();
                for key in chunk {
                    scratch.push(map.get(black_box(*key)));
                }
                black_box(&scratch);
            }
        });
    });

    group.bench_function("elastic_multi_get_into", |b| {
        let map = build_elastic_map(&pairs);
        let mut scratch: Vec<Option<&u64>> = Vec::with_capacity(GET_MANY_BATCH);
        b.iter(|| {
            for chunk in &chunks {
                map.multi_get_into(chunk, &mut scratch);
                black_box(&scratch);
            }
        });
    });

    group.bench_function("funnel_multi_get_into", |b| {
        let map = build_funnel_map(&pairs);
        let mut scratch: Vec<Option<&u64>> = Vec::with_capacity(GET_MANY_BATCH);
        b.iter(|| {
            for chunk in &chunks {
                map.multi_get_into(chunk, &mut scratch);
                black_box(&scratch);
            }
        });
    });

    group.finish();
}

fn bench_get_hit_latency(c: &mut Criterion) {
    for &size in LATENCY_SIZES {
        let pairs = make_pairs(size);
        let query_keys: Vec<u64> = (0..size).map(|idx| pairs[idx].0).collect();

        let label = size_label(size);
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

        group.bench_function("hashbrown", |b| {
            let map = build_hashbrown_map(&pairs);
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

/// Get-hit throughput at pow2 group_count capacities (triangular path fires).
/// Compare to `bench_get_hit_throughput` (non-pow2) to isolate the algorithmic
/// win from the dispatch refactor.
fn bench_pow2_lookup_throughput(c: &mut Criterion) {
    let elastic_pairs = make_pairs(LOOKUP_MAP_SIZE_POW2);
    let elastic_keys: Vec<u64> = (0..HIT_LOOKUP_COUNT)
        .map(|idx| elastic_pairs[idx % LOOKUP_MAP_SIZE_POW2].0)
        .collect();

    let funnel_pairs = make_pairs(FUNNEL_POW2_MAP_SIZE);
    let funnel_keys: Vec<u64> = (0..HIT_LOOKUP_COUNT)
        .map(|idx| funnel_pairs[idx % FUNNEL_POW2_MAP_SIZE].0)
        .collect();

    let mut group = c.benchmark_group("get_hit_throughput_pow2");
    group.throughput(Throughput::Elements(HIT_LOOKUP_COUNT as u64));

    group.bench_function("elastic", |b| {
        b.iter_batched(
            || build_elastic_map(&elastic_pairs),
            |map| {
                for key in &elastic_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("funnel", |b| {
        b.iter_batched(
            || build_funnel_map(&funnel_pairs),
            |map| {
                for key in &funnel_keys {
                    black_box(map.get(black_box(key)));
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .with_profiler(FlamegraphProfiler::new())
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets =
        bench_insert_throughput,
        bench_get_hit_throughput,
        bench_get_miss_throughput,
        bench_tiny_lookup_throughput,
        bench_delete_heavy_throughput,
        bench_resize_heavy_throughput,
        bench_mixed_lookup_throughput,
        bench_multi_get_batch,
        bench_pow2_lookup_throughput,
        bench_get_hit_latency
);
criterion_main!(benches);

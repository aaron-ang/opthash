use std::collections::HashMap as StdHashMap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use opthash::{ElasticHashMap, FunnelHashMap};
use opthash_internal::{
    CONTROL_GROUP_SIZE, CTRL_EMPTY, CTRL_TOMBSTONE, control_match_fingerprint_group,
    control_match_free_group, preferred_group_width,
};

const CONTROL_SCAN_COUNT: usize = 1_024;
const CONTROL_GROUP_OPS: usize = 4_096;
const MAP_SIZE: usize = 20_000;
const MISS_LOOKUP_COUNT: usize = 100_000;
const RESIZE_TRIGGER_COUNT: usize = 2_000;
const FUNNEL_DELETE_CHURN_COUNT: usize = 6_000;
const SPECIAL_LOOKUP_CAPACITY: usize = 512;
const SPECIAL_LOOKUP_LOAD_NUMERATOR: usize = 3;
const SPECIAL_LOOKUP_LOAD_DENOMINATOR: usize = 4;
const SPECIAL_LOOKUP_COUNT: usize = 200_000;

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

fn build_funnel_map(pairs: &[(u64, u64)]) -> FunnelHashMap<u64, u64> {
    let mut map = FunnelHashMap::with_capacity(pairs.len() * 2);
    for &(key, value) in pairs {
        map.insert(key, value);
    }
    map
}

fn control_fingerprint(hash: u64) -> u8 {
    let low = u8::try_from(hash & 0x7F).expect("7-bit fingerprint fits in u8");
    low.max(1)
}

fn fingerprint_bit(fingerprint: u8) -> u128 {
    1u128 << u32::from(fingerprint - 1)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn log_log_probe_limit(capacity: usize) -> usize {
    let n = capacity.max(4) as f64;
    n.log2().max(2.0).log2().ceil().max(1.0) as usize
}

#[allow(clippy::cast_possible_truncation)]
fn hash_to_usize(hash: u64) -> usize {
    hash as usize
}

fn greatest_common_divisor(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn build_group_steps(group_count: usize) -> Box<[usize]> {
    if group_count <= 1 {
        return Box::new([1]);
    }

    let mut steps = Vec::new();
    for step in 1..group_count {
        if greatest_common_divisor(step, group_count) == 1 {
            steps.push(step);
        }
    }
    if steps.is_empty() {
        steps.push(1);
    }
    steps.into_boxed_slice()
}

fn advance_wrapping_index(index: usize, step: usize, len: usize) -> usize {
    if len == 0 { 0 } else { (index + step) % len }
}

fn find_first_free_in_controls(controls: &[u8]) -> Option<usize> {
    if controls.len() < CONTROL_GROUP_SIZE {
        return controls
            .iter()
            .position(|&control| control == CTRL_EMPTY || control == CTRL_TOMBSTONE);
    }

    let wide = preferred_group_width();
    let mut index = 0usize;
    while wide > CONTROL_GROUP_SIZE && index + wide <= controls.len() {
        let mask = control_match_free_group(&controls[index..index + wide]);
        if mask != 0 {
            return Some(index + mask.trailing_zeros() as usize);
        }
        index += wide;
    }

    while index + CONTROL_GROUP_SIZE <= controls.len() {
        let mask = control_match_free_group(&controls[index..index + CONTROL_GROUP_SIZE]);
        if mask != 0 {
            return Some(index + mask.trailing_zeros() as usize);
        }
        index += CONTROL_GROUP_SIZE;
    }

    controls[index..]
        .iter()
        .position(|&control| control == CTRL_EMPTY || control == CTRL_TOMBSTONE)
        .map(|offset| index + offset)
}

fn find_next_fingerprint(controls: &[u8], fingerprint: u8, start: usize) -> Option<usize> {
    if start >= controls.len() {
        return None;
    }

    if controls.len() - start < CONTROL_GROUP_SIZE {
        return controls[start..]
            .iter()
            .position(|&control| control == fingerprint)
            .map(|offset| start + offset);
    }

    let wide = preferred_group_width();
    let mut index = start;
    while wide > CONTROL_GROUP_SIZE && index + wide <= controls.len() {
        let mask = control_match_fingerprint_group(&controls[index..index + wide], fingerprint);
        if mask != 0 {
            return Some(index + mask.trailing_zeros() as usize);
        }
        index += wide;
    }

    while index + CONTROL_GROUP_SIZE <= controls.len() {
        let mask = control_match_fingerprint_group(
            &controls[index..index + CONTROL_GROUP_SIZE],
            fingerprint,
        );
        if mask != 0 {
            return Some(index + mask.trailing_zeros() as usize);
        }
        index += CONTROL_GROUP_SIZE;
    }

    controls[index..]
        .iter()
        .position(|&control| control == fingerprint)
        .map(|offset| index + offset)
}

#[derive(Debug)]
struct SpecialPrimaryFixture {
    controls: Vec<u8>,
    keys: Vec<u64>,
    group_summaries: Box<[u128]>,
    group_steps: Box<[usize]>,
    probe_limit: usize,
    hash_builder: RandomState,
}

impl SpecialPrimaryFixture {
    fn new(capacity: usize) -> Self {
        let group_count = capacity.div_ceil(CONTROL_GROUP_SIZE);
        Self {
            controls: vec![CTRL_EMPTY; capacity],
            keys: vec![0; capacity],
            group_summaries: vec![0; group_count].into_boxed_slice(),
            group_steps: build_group_steps(group_count),
            probe_limit: log_log_probe_limit(capacity),
            hash_builder: RandomState::new(),
        }
    }

    fn hash_key<T>(&self, key: &T) -> u64
    where
        T: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    fn probe_params(&self, key_hash: u64) -> (usize, usize) {
        let group_count = self.group_summaries.len().max(1);
        if group_count <= 1 {
            return (0, 1);
        }
        let start = hash_to_usize(key_hash.rotate_left(11)) % group_count;
        let step =
            self.group_steps[hash_to_usize(key_hash.rotate_left(43)) % self.group_steps.len()];
        (start, step)
    }

    fn group_range(&self, group_idx: usize) -> std::ops::Range<usize> {
        let start = group_idx * CONTROL_GROUP_SIZE;
        let end = (start + CONTROL_GROUP_SIZE).min(self.controls.len());
        start..end
    }

    fn insert(&mut self, key: u64) -> bool {
        let key_hash = self.hash_key(&key);
        let fingerprint = control_fingerprint(key_hash);
        let (group_start, group_step) = self.probe_params(key_hash);
        let group_count = self.group_summaries.len().max(1);
        let mut group_idx = group_start;
        let group_limit = self.probe_limit.min(group_count);

        for _ in 0..group_limit {
            let group_range = self.group_range(group_idx);
            if let Some(offset) = find_first_free_in_controls(&self.controls[group_range.clone()]) {
                let slot_idx = group_range.start + offset;
                self.controls[slot_idx] = fingerprint;
                self.keys[slot_idx] = key;
                self.group_summaries[group_idx] |= fingerprint_bit(fingerprint);
                return true;
            }
            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }

        false
    }

    fn contains(&self, key: u64) -> bool {
        let key_hash = self.hash_key(&key);
        let fingerprint = control_fingerprint(key_hash);
        let fingerprint_mask = fingerprint_bit(fingerprint);
        let (group_start, group_step) = self.probe_params(key_hash);
        let group_count = self.group_summaries.len().max(1);
        let mut group_idx = group_start;
        let group_limit = self.probe_limit.min(group_count);

        for _ in 0..group_limit {
            if self.group_summaries[group_idx] & fingerprint_mask != 0 {
                let group_range = self.group_range(group_idx);
                let controls = &self.controls[group_range.clone()];
                let mut offset = 0usize;
                while let Some(relative_idx) = find_next_fingerprint(controls, fingerprint, offset)
                {
                    let slot_idx = group_range.start + relative_idx;
                    if self.keys[slot_idx] == key {
                        return true;
                    }
                    offset = relative_idx + 1;
                }
            }

            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }

        false
    }
}

#[derive(Debug)]
struct PaperCloserPrimaryFixture {
    controls: Vec<u8>,
    keys: Vec<u64>,
    probe_limit: usize,
    hash_builder: RandomState,
}

impl PaperCloserPrimaryFixture {
    fn new(capacity: usize) -> Self {
        Self {
            controls: vec![CTRL_EMPTY; capacity],
            keys: vec![0; capacity],
            probe_limit: log_log_probe_limit(capacity),
            hash_builder: RandomState::new(),
        }
    }

    fn hash_key<T>(&self, key: &T) -> u64
    where
        T: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    fn probe_params(&self, key_hash: u64) -> (usize, usize) {
        let capacity = self.controls.len().max(1);
        if capacity <= 1 {
            return (0, 1);
        }
        let start = hash_to_usize(key_hash.rotate_left(11)) % capacity;
        let mut step = (hash_to_usize(key_hash.rotate_left(43)) % capacity).max(1);
        while greatest_common_divisor(step, capacity) != 1 {
            step += 1;
            if step >= capacity {
                step = 1;
            }
        }
        (start, step)
    }

    fn insert(&mut self, key: u64) -> bool {
        let key_hash = self.hash_key(&key);
        let fingerprint = control_fingerprint(key_hash);
        let (mut slot_idx, step) = self.probe_params(key_hash);

        for _ in 0..self.probe_limit.min(self.controls.len().max(1)) {
            let control = self.controls[slot_idx];
            if control == CTRL_EMPTY || control == CTRL_TOMBSTONE {
                self.controls[slot_idx] = fingerprint;
                self.keys[slot_idx] = key;
                return true;
            }
            slot_idx = advance_wrapping_index(slot_idx, step, self.controls.len());
        }

        false
    }

    fn contains(&self, key: u64) -> bool {
        let key_hash = self.hash_key(&key);
        let fingerprint = control_fingerprint(key_hash);
        let (mut slot_idx, step) = self.probe_params(key_hash);

        for _ in 0..self.probe_limit.min(self.controls.len().max(1)) {
            let control = self.controls[slot_idx];
            if control == CTRL_EMPTY {
                return false;
            }
            if control == fingerprint && self.keys[slot_idx] == key {
                return true;
            }
            slot_idx = advance_wrapping_index(slot_idx, step, self.controls.len());
        }

        false
    }
}

#[derive(Debug)]
struct SpecialFallbackFixture {
    controls: Vec<u8>,
    keys: Vec<u64>,
    bucket_size: usize,
    bucket_summaries: Box<[u128]>,
    hash_builder: RandomState,
}

impl SpecialFallbackFixture {
    fn new(capacity: usize, primary_probe_limit: usize) -> Self {
        let bucket_size = (2 * primary_probe_limit).max(2);
        let bucket_count = capacity.div_ceil(bucket_size);
        Self {
            controls: vec![CTRL_EMPTY; capacity],
            keys: vec![0; capacity],
            bucket_size,
            bucket_summaries: vec![0; bucket_count].into_boxed_slice(),
            hash_builder: RandomState::new(),
        }
    }

    fn hash_key<T>(&self, key: &T) -> u64
    where
        T: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    fn bucket_count(&self) -> usize {
        self.bucket_summaries.len().max(1)
    }

    fn bucket_a(&self, key_hash: u64) -> usize {
        hash_to_usize(key_hash.rotate_left(19)) % self.bucket_count()
    }

    fn bucket_b(&self, key_hash: u64) -> usize {
        hash_to_usize(key_hash.rotate_left(37)) % self.bucket_count()
    }

    fn bucket_range(&self, bucket_idx: usize) -> std::ops::Range<usize> {
        let start = bucket_idx * self.bucket_size;
        let end = (start + self.bucket_size).min(self.controls.len());
        start..end
    }

    fn bucket_len(&self, bucket_idx: usize) -> usize {
        self.bucket_range(bucket_idx).len()
    }

    fn insert(&mut self, key: u64) -> bool {
        let key_hash = self.hash_key(&key);
        let fingerprint = control_fingerprint(key_hash);
        let bucket_a = self.bucket_a(key_hash);
        let bucket_b = self.bucket_b(key_hash);
        let max_bucket_len = self.bucket_len(bucket_a).max(self.bucket_len(bucket_b));

        for offset in 0..max_bucket_len {
            if offset < self.bucket_len(bucket_a) {
                let slot_idx = bucket_a * self.bucket_size + offset;
                let control = self.controls[slot_idx];
                if control == CTRL_EMPTY || control == CTRL_TOMBSTONE {
                    self.controls[slot_idx] = fingerprint;
                    self.keys[slot_idx] = key;
                    self.bucket_summaries[bucket_a] |= fingerprint_bit(fingerprint);
                    return true;
                }
            }
            if bucket_b != bucket_a && offset < self.bucket_len(bucket_b) {
                let slot_idx = bucket_b * self.bucket_size + offset;
                let control = self.controls[slot_idx];
                if control == CTRL_EMPTY || control == CTRL_TOMBSTONE {
                    self.controls[slot_idx] = fingerprint;
                    self.keys[slot_idx] = key;
                    self.bucket_summaries[bucket_b] |= fingerprint_bit(fingerprint);
                    return true;
                }
            }
        }

        false
    }

    fn contains(&self, key: u64) -> bool {
        let key_hash = self.hash_key(&key);
        let fingerprint = control_fingerprint(key_hash);
        let fingerprint_mask = fingerprint_bit(fingerprint);
        let bucket_a = self.bucket_a(key_hash);
        let bucket_b = self.bucket_b(key_hash);

        for bucket_idx in [bucket_a, bucket_b] {
            if self.bucket_summaries[bucket_idx] & fingerprint_mask == 0 {
                continue;
            }

            let range = self.bucket_range(bucket_idx);
            let controls = &self.controls[range.clone()];
            let mut offset = 0usize;
            while let Some(relative_idx) = find_next_fingerprint(controls, fingerprint, offset) {
                let slot_idx = range.start + relative_idx;
                if self.keys[slot_idx] == key {
                    return true;
                }
                offset = relative_idx + 1;
            }
        }

        false
    }
}

fn build_special_primary_fixture() -> (SpecialPrimaryFixture, Vec<u64>) {
    let insert_count =
        (SPECIAL_LOOKUP_CAPACITY * SPECIAL_LOOKUP_LOAD_NUMERATOR) / SPECIAL_LOOKUP_LOAD_DENOMINATOR;
    let mut fixture = SpecialPrimaryFixture::new(SPECIAL_LOOKUP_CAPACITY);
    let mut inserted_keys = Vec::with_capacity(insert_count);
    let mut candidate_idx = 0usize;

    while inserted_keys.len() < insert_count {
        let key = key_at(candidate_idx + 1_000_000);
        candidate_idx += 1;
        if fixture.insert(key) {
            inserted_keys.push(key);
        }
    }

    (fixture, inserted_keys)
}

fn build_paper_closer_primary_fixture() -> (PaperCloserPrimaryFixture, Vec<u64>) {
    let insert_count =
        (SPECIAL_LOOKUP_CAPACITY * SPECIAL_LOOKUP_LOAD_NUMERATOR) / SPECIAL_LOOKUP_LOAD_DENOMINATOR;
    let mut fixture = PaperCloserPrimaryFixture::new(SPECIAL_LOOKUP_CAPACITY);
    let mut inserted_keys = Vec::with_capacity(insert_count);
    let mut candidate_idx = 0usize;

    while inserted_keys.len() < insert_count {
        let key = key_at(candidate_idx + 2_000_000);
        candidate_idx += 1;
        if fixture.insert(key) {
            inserted_keys.push(key);
        }
    }

    (fixture, inserted_keys)
}

fn build_special_fallback_fixture() -> (SpecialFallbackFixture, Vec<u64>) {
    let primary_probe_limit = log_log_probe_limit(SPECIAL_LOOKUP_CAPACITY);
    let insert_count =
        (SPECIAL_LOOKUP_CAPACITY * SPECIAL_LOOKUP_LOAD_NUMERATOR) / SPECIAL_LOOKUP_LOAD_DENOMINATOR;
    let mut fixture = SpecialFallbackFixture::new(SPECIAL_LOOKUP_CAPACITY, primary_probe_limit);
    let mut inserted_keys = Vec::with_capacity(insert_count);
    let mut candidate_idx = 0usize;

    while inserted_keys.len() < insert_count {
        let key = key_at(candidate_idx + 3_000_000);
        candidate_idx += 1;
        if fixture.insert(key) {
            inserted_keys.push(key);
        }
    }

    (fixture, inserted_keys)
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

fn bench_funnel_bucket_lookup(c: &mut Criterion) {
    let pairs = make_pairs(MAP_SIZE);
    let trailing_live_keys: Vec<u64> = pairs[..(MAP_SIZE * 3 / 4)]
        .iter()
        .map(|&(key, _)| key)
        .collect();
    let dense_live_keys: Vec<u64> = pairs
        .iter()
        .enumerate()
        .filter(|(idx, _)| idx % 3 != 0)
        .map(|(_, &(key, _))| key)
        .collect();

    let mut group = c.benchmark_group("internal_funnel_bucket_lookup");
    group.throughput(Throughput::Elements(MISS_LOOKUP_COUNT as u64));

    group.bench_function("no_tombstones", |b| {
        b.iter_batched(
            || build_funnel_map(&pairs),
            |map| {
                for idx in 0..MISS_LOOKUP_COUNT {
                    let key = pairs[idx % pairs.len()].0;
                    black_box(map.get(black_box(&key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("trailing_tombstones", |b| {
        b.iter_batched(
            || {
                let mut map = build_funnel_map(&pairs);
                for &(key, _) in &pairs[(MAP_SIZE * 3 / 4)..] {
                    black_box(map.remove(black_box(&key)));
                }
                map
            },
            |map| {
                for idx in 0..MISS_LOOKUP_COUNT {
                    let key = trailing_live_keys[idx % trailing_live_keys.len()];
                    black_box(map.get(black_box(&key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("dense_tombstones", |b| {
        b.iter_batched(
            || {
                let mut map = build_funnel_map(&pairs);
                for (idx, &(key, _)) in pairs.iter().enumerate() {
                    if idx % 3 == 0 {
                        black_box(map.remove(black_box(&key)));
                    }
                }
                map
            },
            |map| {
                for idx in 0..MISS_LOOKUP_COUNT {
                    let key = dense_live_keys[idx % dense_live_keys.len()];
                    black_box(map.get(black_box(&key)));
                }
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn bench_funnel_delete_churn(c: &mut Criterion) {
    let pairs = make_pairs(FUNNEL_DELETE_CHURN_COUNT);
    let mut group = c.benchmark_group("internal_funnel_delete_churn");
    group.throughput(Throughput::Elements((FUNNEL_DELETE_CHURN_COUNT * 2) as u64));
    group.bench_function("funnel", |b| {
        b.iter_batched(
            || build_funnel_map(&pairs),
            |mut map| {
                for &(key, value) in &pairs {
                    black_box(map.remove(black_box(&key)));
                    black_box(map.insert(black_box(key), black_box(value)));
                }
                black_box(map.len())
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_funnel_special_lookup(c: &mut Criterion) {
    let (special_primary, special_primary_keys) = build_special_primary_fixture();
    let (paper_closer_primary, paper_closer_primary_keys) = build_paper_closer_primary_fixture();
    let (special_fallback, special_fallback_keys) = build_special_fallback_fixture();

    let mut group = c.benchmark_group("internal_funnel_special_lookup");
    group.throughput(Throughput::Elements(SPECIAL_LOOKUP_COUNT as u64));

    group.bench_function("special_primary_hits", |b| {
        b.iter(|| {
            for idx in 0..SPECIAL_LOOKUP_COUNT {
                let key = special_primary_keys[idx % special_primary_keys.len()];
                black_box(special_primary.contains(black_box(key)));
            }
        });
    });

    group.bench_function("paper_closer_b_hits", |b| {
        b.iter(|| {
            for idx in 0..SPECIAL_LOOKUP_COUNT {
                let key = paper_closer_primary_keys[idx % paper_closer_primary_keys.len()];
                black_box(paper_closer_primary.contains(black_box(key)));
            }
        });
    });

    group.bench_function("special_fallback_hits", |b| {
        b.iter(|| {
            for idx in 0..SPECIAL_LOOKUP_COUNT {
                let key = special_fallback_keys[idx % special_fallback_keys.len()];
                black_box(special_fallback.contains(black_box(key)));
            }
        });
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
        bench_funnel_bucket_lookup,
        bench_funnel_delete_churn,
        bench_funnel_special_lookup,
        bench_resize_cost
);
criterion_main!(benches);

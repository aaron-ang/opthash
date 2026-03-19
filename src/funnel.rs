use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

use crate::common::{
    config::{DEFAULT_RESERVE_FRACTION, INITIAL_CAPACITY},
    control::{CTRL_EMPTY, ControlByte, Controls, control_fingerprint, fingerprint_bit},
    layout::{Entry, GROUP_SIZE, RawTable},
    math::{
        advance_wrapping_index, greatest_common_divisor, round_up_to_group,
        sanitize_reserve_fraction,
    },
};

const MAX_FUNNEL_RESERVE_FRACTION: f64 = 1.0 / 8.0;

#[derive(Debug, Clone)]
pub struct FunnelOptions {
    pub capacity: usize,
    pub reserve_fraction: f64,
    pub primary_probe_limit: Option<usize>,
}

impl Default for FunnelOptions {
    fn default() -> Self {
        Self {
            capacity: 0,
            reserve_fraction: DEFAULT_RESERVE_FRACTION,
            primary_probe_limit: None,
        }
    }
}

impl FunnelOptions {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            ..Self::default()
        }
    }
}

#[derive(Debug)]
struct BucketLevel<K, V> {
    table: RawTable<Entry<K, V>>,
    len: usize,
    bucket_size: usize,
    bucket_summaries: Box<[u128]>,
    bucket_live: Box<[usize]>,
    bucket_tombstones: Box<[usize]>,
}

impl<K, V> BucketLevel<K, V> {
    fn with_bucket_count(bucket_count: usize, bucket_size: usize) -> Self {
        let total_capacity = bucket_count.saturating_mul(bucket_size);
        Self {
            table: RawTable::new(total_capacity),
            len: 0,
            bucket_size,
            bucket_summaries: vec![0; bucket_count].into_boxed_slice(),
            bucket_live: vec![0; bucket_count].into_boxed_slice(),
            bucket_tombstones: vec![0; bucket_count].into_boxed_slice(),
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.table.capacity()
    }

    #[inline]
    fn bucket_count(&self) -> usize {
        if self.bucket_size == 0 {
            0
        } else {
            self.table.capacity() / self.bucket_size
        }
    }

    #[inline]
    fn bucket_range(&self, bucket_idx: usize) -> std::ops::Range<usize> {
        let start = bucket_idx * self.bucket_size;
        start..start + self.bucket_size
    }
}

impl<K, V> Drop for BucketLevel<K, V> {
    fn drop(&mut self) {
        for idx in 0..self.table.capacity() {
            if self.table.control_at(idx).is_occupied() {
                unsafe { self.table.drop_in_place(idx) };
            }
        }
    }
}

#[derive(Debug)]
struct SpecialPrimary<K, V> {
    table: RawTable<Entry<K, V>>,
    len: usize,
    group_summaries: Box<[u128]>,
    group_tombstones: Box<[usize]>,
    group_steps: Box<[usize]>,
}

impl<K, V> SpecialPrimary<K, V> {
    fn with_capacity(capacity: usize) -> Self {
        let table = RawTable::new(capacity);
        let group_count = table.group_count();
        Self {
            table,
            len: 0,
            group_summaries: vec![0; group_count].into_boxed_slice(),
            group_tombstones: vec![0; group_count].into_boxed_slice(),
            group_steps: build_group_steps(group_count),
        }
    }
}

impl<K, V> Drop for SpecialPrimary<K, V> {
    fn drop(&mut self) {
        for idx in 0..self.table.capacity() {
            if self.table.control_at(idx).is_occupied() {
                unsafe { self.table.drop_in_place(idx) };
            }
        }
    }
}

#[derive(Debug)]
struct SpecialFallback<K, V> {
    table: RawTable<Entry<K, V>>,
    len: usize,
    bucket_size: usize,
    bucket_summaries: Box<[u128]>,
    bucket_live: Box<[usize]>,
    bucket_tombstones: Box<[usize]>,
}

impl<K, V> SpecialFallback<K, V> {
    fn with_capacity(capacity: usize, bucket_size: usize) -> Self {
        let bucket_count = if bucket_size == 0 {
            0
        } else {
            capacity / bucket_size
        };
        Self {
            table: RawTable::new(capacity),
            len: 0,
            bucket_size,
            bucket_summaries: vec![0; bucket_count].into_boxed_slice(),
            bucket_live: vec![0; bucket_count].into_boxed_slice(),
            bucket_tombstones: vec![0; bucket_count].into_boxed_slice(),
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.table.capacity()
    }

    #[inline]
    fn bucket_count(&self) -> usize {
        if self.bucket_size == 0 {
            0
        } else {
            self.table.capacity() / self.bucket_size
        }
    }

    #[inline]
    fn bucket_range(&self, bucket_idx: usize) -> std::ops::Range<usize> {
        let start = bucket_idx * self.bucket_size;
        start..start + self.bucket_size
    }
}

impl<K, V> Drop for SpecialFallback<K, V> {
    fn drop(&mut self) {
        for idx in 0..self.table.capacity() {
            if self.table.control_at(idx).is_occupied() {
                unsafe { self.table.drop_in_place(idx) };
            }
        }
    }
}

#[derive(Debug)]
struct SpecialArray<K, V> {
    primary: SpecialPrimary<K, V>,
    fallback: SpecialFallback<K, V>,
}

impl<K, V> SpecialArray<K, V> {
    fn with_capacity(capacity: usize, primary_probe_limit: usize) -> Self {
        let fallback_bucket_size =
            round_up_to_group((2usize.saturating_mul(primary_probe_limit)).max(2));
        let desired_fallback_capacity = capacity / 2;
        let fallback_bucket_count = desired_fallback_capacity / fallback_bucket_size.max(1);
        let fallback_capacity = fallback_bucket_count.saturating_mul(fallback_bucket_size);
        let primary_capacity = capacity.saturating_sub(fallback_capacity);
        Self {
            primary: SpecialPrimary::with_capacity(primary_capacity),
            fallback: SpecialFallback::with_capacity(fallback_capacity, fallback_bucket_size),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotLocation {
    Level { level_idx: usize, slot_idx: usize },
    SpecialPrimary { slot_idx: usize },
    SpecialFallback { slot_idx: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LookupStep {
    Found(usize),
    Continue,
    StopSearch,
}

#[derive(Debug)]
pub struct FunnelHashMap<K, V> {
    levels: Vec<BucketLevel<K, V>>,
    special: SpecialArray<K, V>,
    len: usize,
    capacity: usize,
    max_insertions: usize,
    reserve_fraction: f64,
    primary_probe_limit: usize,
    max_populated_level: usize,
    hash_builder: RandomState,
}

impl<K, V> Default for FunnelHashMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> FunnelHashMap<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self::with_options(FunnelOptions::default())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_options(FunnelOptions::with_capacity(capacity))
    }

    pub fn with_options(options: FunnelOptions) -> Self {
        Self::with_options_and_hasher(options, RandomState::new())
    }

    fn with_options_and_hasher(options: FunnelOptions, hash_builder: RandomState) -> Self {
        let reserve_fraction =
            sanitize_reserve_fraction(options.reserve_fraction).min(MAX_FUNNEL_RESERVE_FRACTION);
        let capacity = options.capacity;
        let max_insertions =
            capacity.saturating_sub((reserve_fraction * capacity as f64).floor() as usize);

        let level_count = compute_level_count(reserve_fraction);
        let bucket_width = round_up_to_group(compute_bucket_width(reserve_fraction));
        let primary_probe_limit = options
            .primary_probe_limit
            .unwrap_or_else(|| log_log_probe_limit(capacity))
            .max(1);

        let mut special_capacity =
            choose_special_capacity(capacity, reserve_fraction, bucket_width);
        let mut main_capacity = capacity.saturating_sub(special_capacity);
        let main_remainder = main_capacity % bucket_width.max(1);
        if main_remainder != 0 {
            main_capacity = main_capacity.saturating_sub(main_remainder);
            special_capacity = capacity.saturating_sub(main_capacity);
        }

        let total_main_buckets = if bucket_width == 0 {
            0
        } else {
            main_capacity / bucket_width
        };
        let level_bucket_counts = partition_funnel_buckets(total_main_buckets, level_count);
        let levels = level_bucket_counts
            .into_iter()
            .map(|bucket_count| BucketLevel::with_bucket_count(bucket_count, bucket_width))
            .collect::<Vec<_>>();

        let special = SpecialArray::with_capacity(special_capacity, primary_probe_limit);

        Self {
            levels,
            special,
            len: 0,
            capacity,
            max_insertions,
            reserve_fraction,
            primary_probe_limit,
            max_populated_level: 0,
            hash_builder,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let key_hash = self.hash_key(&key);
        let key_fingerprint = control_fingerprint(key_hash);

        if let Some(location) = self.find_slot_location_with_hash(&key, key_hash, key_fingerprint) {
            return Some(self.replace_existing_value(location, value));
        }

        if self.len >= self.max_insertions {
            let new_capacity = if self.capacity == 0 {
                INITIAL_CAPACITY
            } else {
                self.capacity.saturating_mul(2)
            };
            self.resize(new_capacity);
        }

        let insertion_slot = self
            .choose_slot_for_new_key(key_hash)
            .expect("no free slot found after resize");
        self.place_new_entry(insertion_slot, key, value, key_fingerprint);
        None
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = control_fingerprint(key_hash);

        match self.find_slot_location_with_hash(key, key_hash, key_fingerprint)? {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => Some(unsafe { &self.levels[level_idx].table.get_ref(slot_idx).value }),
            SlotLocation::SpecialPrimary { slot_idx } => {
                Some(unsafe { &self.special.primary.table.get_ref(slot_idx).value })
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                Some(unsafe { &self.special.fallback.table.get_ref(slot_idx).value })
            }
        }
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = control_fingerprint(key_hash);

        match self.find_slot_location_with_hash(key, key_hash, key_fingerprint)? {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => Some(unsafe { &mut self.levels[level_idx].table.get_mut(slot_idx).value }),
            SlotLocation::SpecialPrimary { slot_idx } => {
                Some(unsafe { &mut self.special.primary.table.get_mut(slot_idx).value })
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                Some(unsafe { &mut self.special.fallback.table.get_mut(slot_idx).value })
            }
        }
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = control_fingerprint(key_hash);
        self.find_slot_location_with_hash(key, key_hash, key_fingerprint)
            .is_some()
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = control_fingerprint(key_hash);
        let location = self.find_slot_location_with_hash(key, key_hash, key_fingerprint)?;

        let removed_entry = match location {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => {
                let bucket_idx = self.level_bucket_index(
                    key_hash,
                    level_idx,
                    self.levels[level_idx].bucket_count(),
                );
                let level = &mut self.levels[level_idx];
                let removed = unsafe { level.table.take(slot_idx) };
                level.table.mark_tombstone(slot_idx);
                level.len -= 1;
                level.bucket_live[bucket_idx] -= 1;
                level.bucket_tombstones[bucket_idx] += 1;
                Self::rebuild_bucket_summary(
                    &level.table,
                    level.bucket_range(bucket_idx),
                    &mut level.bucket_summaries[bucket_idx],
                );
                if level.bucket_tombstones[bucket_idx] > level.bucket_size / 4 {
                    self.rebuild_level_bucket(level_idx, bucket_idx);
                }
                removed
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                let group_idx = slot_idx / GROUP_SIZE;
                let removed = {
                    let primary = &mut self.special.primary;
                    let removed = unsafe { primary.table.take(slot_idx) };
                    primary.table.mark_tombstone(slot_idx);
                    primary.len -= 1;
                    primary.group_tombstones[group_idx] += 1;
                    removed
                };
                self.rebuild_primary_group_summary(group_idx);
                if self.special.primary.group_tombstones[group_idx]
                    > self.special.primary.table.group_len(group_idx) / 4
                {
                    self.rebuild_special_primary_group(group_idx);
                }
                removed
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                let bucket_idx = self.special_fallback_bucket_of_slot(slot_idx);
                let fallback = &mut self.special.fallback;
                let removed = unsafe { fallback.table.take(slot_idx) };
                fallback.table.mark_tombstone(slot_idx);
                fallback.len -= 1;
                fallback.bucket_live[bucket_idx] -= 1;
                fallback.bucket_tombstones[bucket_idx] += 1;
                Self::rebuild_bucket_summary(
                    &fallback.table,
                    fallback.bucket_range(bucket_idx),
                    &mut fallback.bucket_summaries[bucket_idx],
                );
                if fallback.bucket_tombstones[bucket_idx] > fallback.bucket_size / 4 {
                    self.rebuild_special_fallback_bucket(bucket_idx);
                }
                removed
            }
        };

        self.len -= 1;
        self.shrink_max_populated_level();
        Some(removed_entry.value)
    }

    pub fn clear(&mut self) {
        for level in &mut self.levels {
            for idx in 0..level.table.capacity() {
                if level.table.control_at(idx).is_occupied() {
                    unsafe { level.table.drop_in_place(idx) };
                }
            }
            level.table.clear_all_controls();
            level.len = 0;
            level.bucket_summaries.fill(0);
            level.bucket_live.fill(0);
            level.bucket_tombstones.fill(0);
        }

        for idx in 0..self.special.primary.table.capacity() {
            if self.special.primary.table.control_at(idx).is_occupied() {
                unsafe { self.special.primary.table.drop_in_place(idx) };
            }
        }
        self.special.primary.table.clear_all_controls();
        self.special.primary.len = 0;
        self.special.primary.group_summaries.fill(0);
        self.special.primary.group_tombstones.fill(0);

        for idx in 0..self.special.fallback.table.capacity() {
            if self.special.fallback.table.control_at(idx).is_occupied() {
                unsafe { self.special.fallback.table.drop_in_place(idx) };
            }
        }
        self.special.fallback.table.clear_all_controls();
        self.special.fallback.len = 0;
        self.special.fallback.bucket_summaries.fill(0);
        self.special.fallback.bucket_live.fill(0);
        self.special.fallback.bucket_tombstones.fill(0);

        self.len = 0;
        self.max_populated_level = 0;
    }

    fn resize(&mut self, new_capacity: usize) {
        let mut entries = Vec::with_capacity(self.len);

        for level in &mut self.levels {
            for idx in 0..level.table.capacity() {
                if level.table.control_at(idx).is_occupied() {
                    let entry = unsafe { level.table.take(idx) };
                    entries.push((entry.key, entry.value));
                }
            }
            level.table.clear_all_controls();
            level.len = 0;
            level.bucket_summaries.fill(0);
            level.bucket_live.fill(0);
            level.bucket_tombstones.fill(0);
        }

        for idx in 0..self.special.primary.table.capacity() {
            if self.special.primary.table.control_at(idx).is_occupied() {
                let entry = unsafe { self.special.primary.table.take(idx) };
                entries.push((entry.key, entry.value));
            }
        }
        self.special.primary.table.clear_all_controls();
        self.special.primary.len = 0;
        self.special.primary.group_summaries.fill(0);
        self.special.primary.group_tombstones.fill(0);

        for idx in 0..self.special.fallback.table.capacity() {
            if self.special.fallback.table.control_at(idx).is_occupied() {
                let entry = unsafe { self.special.fallback.table.take(idx) };
                entries.push((entry.key, entry.value));
            }
        }
        self.special.fallback.table.clear_all_controls();
        self.special.fallback.len = 0;
        self.special.fallback.bucket_summaries.fill(0);
        self.special.fallback.bucket_live.fill(0);
        self.special.fallback.bucket_tombstones.fill(0);

        self.len = 0;
        self.max_populated_level = 0;

        let hash_builder = std::mem::take(&mut self.hash_builder);
        let mut new_map = Self::with_options_and_hasher(
            FunnelOptions {
                capacity: new_capacity,
                reserve_fraction: self.reserve_fraction,
                primary_probe_limit: Some(self.primary_probe_limit),
            },
            hash_builder,
        );

        for (key, value) in entries {
            new_map.insert(key, value);
        }

        *self = new_map;
    }

    #[inline]
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    #[inline]
    fn level_bucket_index(&self, key_hash: u64, level_idx: usize, bucket_count: usize) -> usize {
        (key_hash.rotate_left((level_idx as u32 * 7) % 64) as usize) % bucket_count
    }

    #[inline]
    fn special_primary_probe_params(&self, key_hash: u64, group_count: usize) -> (usize, usize) {
        if group_count <= 1 {
            return (0, 1);
        }
        let start = (key_hash.rotate_left(11) as usize) % group_count;
        let steps = &self.special.primary.group_steps;
        let step = steps[(key_hash.rotate_left(43) as usize) % steps.len()];
        (start, step)
    }

    #[inline]
    fn special_fallback_bucket_a(&self, key_hash: u64, bucket_count: usize) -> usize {
        (key_hash.rotate_left(19) as usize) % bucket_count
    }

    #[inline]
    fn special_fallback_bucket_b(&self, key_hash: u64, bucket_count: usize) -> usize {
        (key_hash.rotate_left(37) as usize) % bucket_count
    }

    #[inline]
    fn choose_slot_for_new_key(&self, key_hash: u64) -> Option<SlotLocation> {
        for (level_idx, level) in self.levels.iter().enumerate() {
            if let Some(slot_idx) = self.first_free_in_level_bucket(key_hash, level_idx, level) {
                return Some(SlotLocation::Level {
                    level_idx,
                    slot_idx,
                });
            }
        }

        if let Some(slot_idx) = self.first_free_in_special_primary(key_hash) {
            return Some(SlotLocation::SpecialPrimary { slot_idx });
        }

        self.first_free_in_special_fallback(key_hash)
            .map(|slot_idx| SlotLocation::SpecialFallback { slot_idx })
    }

    #[inline]
    fn replace_existing_value(&mut self, location: SlotLocation, value: V) -> V {
        match location {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => {
                let entry = unsafe { self.levels[level_idx].table.get_mut(slot_idx) };
                std::mem::replace(&mut entry.value, value)
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                let entry = unsafe { self.special.primary.table.get_mut(slot_idx) };
                std::mem::replace(&mut entry.value, value)
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                let entry = unsafe { self.special.fallback.table.get_mut(slot_idx) };
                std::mem::replace(&mut entry.value, value)
            }
        }
    }

    #[inline]
    fn place_new_entry(&mut self, location: SlotLocation, key: K, value: V, key_fingerprint: u8) {
        match location {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => {
                let bucket_idx = slot_idx / self.levels[level_idx].bucket_size;
                let level = &mut self.levels[level_idx];
                level
                    .table
                    .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
                level.len += 1;
                level.bucket_live[bucket_idx] += 1;
                level.bucket_summaries[bucket_idx] |= fingerprint_bit(key_fingerprint);
                if level_idx > self.max_populated_level {
                    self.max_populated_level = level_idx;
                }
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                let group_idx = slot_idx / GROUP_SIZE;
                let primary = &mut self.special.primary;
                primary
                    .table
                    .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
                primary.len += 1;
                primary.group_summaries[group_idx] |= fingerprint_bit(key_fingerprint);
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                let bucket_idx = self.special_fallback_bucket_of_slot(slot_idx);
                let fallback = &mut self.special.fallback;
                fallback
                    .table
                    .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
                fallback.len += 1;
                fallback.bucket_live[bucket_idx] += 1;
                fallback.bucket_summaries[bucket_idx] |= fingerprint_bit(key_fingerprint);
            }
        }
        self.len += 1;
    }

    fn first_free_in_level_bucket(
        &self,
        key_hash: u64,
        level_idx: usize,
        level: &BucketLevel<K, V>,
    ) -> Option<usize> {
        if level.capacity() == 0 || level.len >= level.capacity() {
            return None;
        }

        let bucket_count = level.bucket_count();
        if bucket_count == 0 {
            return None;
        }

        let bucket_idx = self.level_bucket_index(key_hash, level_idx, bucket_count);
        if level.bucket_live[bucket_idx] >= level.bucket_size {
            return None;
        }

        let bucket_range = level.bucket_range(bucket_idx);
        let free_offset = level
            .table
            .controls(bucket_range.clone())
            .find_first_free()?;
        Some(bucket_range.start + free_offset)
    }

    fn first_free_in_special_primary(&self, key_hash: u64) -> Option<usize> {
        let primary = &self.special.primary;
        if primary.table.capacity() == 0 || primary.len >= primary.table.capacity() {
            return None;
        }

        let group_count = primary.table.group_count();
        let (group_start, group_step) = self.special_primary_probe_params(key_hash, group_count);
        let mut group_idx = group_start;
        let group_limit = self.primary_probe_limit.min(group_count.max(1));

        for _ in 0..group_limit {
            if !primary.table.group_meta(group_idx).full
                && let Some(slot_idx) = primary.table.first_free_in_group(group_idx, 0)
            {
                return Some(slot_idx);
            }
            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }
        None
    }

    fn first_free_in_special_fallback(&self, key_hash: u64) -> Option<usize> {
        let fallback = &self.special.fallback;
        if fallback.capacity() == 0 || fallback.len >= fallback.capacity() {
            return None;
        }

        let bucket_count = fallback.bucket_count();
        if bucket_count == 0 {
            return None;
        }

        let bucket_a = self.special_fallback_bucket_a(key_hash, bucket_count);
        let bucket_b = self.special_fallback_bucket_b(key_hash, bucket_count);

        if fallback.bucket_live[bucket_a] < fallback.bucket_size {
            let range = fallback.bucket_range(bucket_a);
            if let Some(offset) = fallback.table.controls(range.clone()).find_first_free() {
                return Some(range.start + offset);
            }
        }

        if fallback.bucket_live[bucket_b] < fallback.bucket_size {
            let range = fallback.bucket_range(bucket_b);
            if let Some(offset) = fallback.table.controls(range.clone()).find_first_free() {
                return Some(range.start + offset);
            }
        }

        None
    }

    fn find_in_level_bucket<Q>(
        &self,
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
        level_idx: usize,
        level: &BucketLevel<K, V>,
    ) -> LookupStep
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if level.len == 0 {
            return LookupStep::Continue;
        }

        let bucket_count = level.bucket_count();
        if bucket_count == 0 {
            return LookupStep::Continue;
        }

        let bucket_idx = self.level_bucket_index(key_hash, level_idx, bucket_count);
        let fingerprint_mask = fingerprint_bit(key_fingerprint);
        if level.bucket_summaries[bucket_idx] & fingerprint_mask == 0 {
            if level.bucket_tombstones[bucket_idx] == 0
                && level.bucket_live[bucket_idx] < level.bucket_size
            {
                return LookupStep::StopSearch;
            }
            return LookupStep::Continue;
        }

        let bucket_range = level.bucket_range(bucket_idx);
        let controls = level.table.controls(bucket_range.clone());
        let searchable_len = if level.bucket_tombstones[bucket_idx] > 0 {
            controls.len()
        } else {
            controls.find_first(CTRL_EMPTY).unwrap_or(controls.len())
        };

        let mut match_offset = 0usize;
        while let Some(relative_idx) =
            controls[..searchable_len].find_next(key_fingerprint, match_offset)
        {
            let slot_idx = bucket_range.start + relative_idx;
            let entry = unsafe { level.table.get_ref(slot_idx) };
            if entry.key.borrow() == key {
                return LookupStep::Found(slot_idx);
            }
            match_offset = relative_idx + 1;
        }

        if level.bucket_tombstones[bucket_idx] == 0 && searchable_len < controls.len() {
            LookupStep::StopSearch
        } else {
            LookupStep::Continue
        }
    }

    fn find_in_special_primary<Q>(&self, key_hash: u64, key_fingerprint: u8, key: &Q) -> LookupStep
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let primary = &self.special.primary;
        if primary.table.capacity() == 0 {
            return LookupStep::Continue;
        }

        let fingerprint_mask = fingerprint_bit(key_fingerprint);
        let group_count = primary.table.group_count();
        let (group_start, group_step) = self.special_primary_probe_params(key_hash, group_count);
        let mut group_idx = group_start;
        let group_limit = self.primary_probe_limit.min(group_count.max(1));

        for _ in 0..group_limit {
            if primary.group_summaries[group_idx] & fingerprint_mask != 0 {
                let mut match_mask = primary.table.group_match_mask(group_idx, key_fingerprint);
                while match_mask != 0 {
                    let relative_idx = match_mask.trailing_zeros() as usize;
                    let slot_idx = primary.table.group_start(group_idx) + relative_idx;
                    let entry = unsafe { primary.table.get_ref(slot_idx) };
                    if entry.key.borrow() == key {
                        return LookupStep::Found(slot_idx);
                    }
                    match_mask &= match_mask - 1;
                }
            }

            if primary.group_tombstones[group_idx] == 0 && !primary.table.group_meta(group_idx).full
            {
                return LookupStep::StopSearch;
            }

            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }

        LookupStep::Continue
    }

    fn find_in_special_fallback<Q>(
        &self,
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
    ) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let fallback = &self.special.fallback;
        if fallback.capacity() == 0 {
            return None;
        }

        let bucket_count = fallback.bucket_count();
        if bucket_count == 0 {
            return None;
        }

        let fingerprint_mask = fingerprint_bit(key_fingerprint);
        let bucket_a = self.special_fallback_bucket_a(key_hash, bucket_count);
        let bucket_b = self.special_fallback_bucket_b(key_hash, bucket_count);

        for bucket_idx in [bucket_a, bucket_b] {
            if fallback.bucket_summaries[bucket_idx] & fingerprint_mask == 0 {
                continue;
            }
            let range = fallback.bucket_range(bucket_idx);
            let controls = fallback.table.controls(range.clone());
            let searchable_len = if fallback.bucket_tombstones[bucket_idx] > 0 {
                controls.len()
            } else {
                controls.find_first(CTRL_EMPTY).unwrap_or(controls.len())
            };

            let mut match_offset = 0usize;
            while let Some(relative_idx) =
                controls[..searchable_len].find_next(key_fingerprint, match_offset)
            {
                let slot_idx = range.start + relative_idx;
                let entry = unsafe { fallback.table.get_ref(slot_idx) };
                if entry.key.borrow() == key {
                    return Some(slot_idx);
                }
                match_offset = relative_idx + 1;
            }
        }

        None
    }

    fn find_slot_location_with_hash<Q>(
        &self,
        key: &Q,
        key_hash: u64,
        key_fingerprint: u8,
    ) -> Option<SlotLocation>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let search_limit = (self.max_populated_level + 1).min(self.levels.len());
        for (level_idx, level) in self.levels[..search_limit].iter().enumerate() {
            match self.find_in_level_bucket(key_hash, key_fingerprint, key, level_idx, level) {
                LookupStep::Found(slot_idx) => {
                    return Some(SlotLocation::Level {
                        level_idx,
                        slot_idx,
                    });
                }
                LookupStep::Continue => {}
                LookupStep::StopSearch => return None,
            }
        }

        match self.find_in_special_primary(key_hash, key_fingerprint, key) {
            LookupStep::Found(slot_idx) => return Some(SlotLocation::SpecialPrimary { slot_idx }),
            LookupStep::Continue => {}
            LookupStep::StopSearch => return None,
        }

        self.find_in_special_fallback(key_hash, key_fingerprint, key)
            .map(|slot_idx| SlotLocation::SpecialFallback { slot_idx })
    }

    fn rebuild_level_bucket(&mut self, level_idx: usize, bucket_idx: usize) {
        let mut entries = Vec::new();
        {
            let level = &mut self.levels[level_idx];
            let range = level.bucket_range(bucket_idx);
            for slot_idx in range.clone() {
                if level.table.control_at(slot_idx).is_occupied() {
                    let entry = unsafe { level.table.take(slot_idx) };
                    entries.push((entry.key, entry.value));
                }
                level.table.clear_control(slot_idx);
            }
            level.bucket_live[bucket_idx] = 0;
            level.bucket_tombstones[bucket_idx] = 0;
            level.bucket_summaries[bucket_idx] = 0;
        }

        for (key, value) in entries {
            let key_hash = self.hash_key(&key);
            let key_fingerprint = control_fingerprint(key_hash);
            let range = self.levels[level_idx].bucket_range(bucket_idx);
            let offset = self.levels[level_idx]
                .table
                .controls(range.clone())
                .find_first_free()
                .expect("rebuilt bucket should have free space");
            let slot_idx = range.start + offset;
            let level = &mut self.levels[level_idx];
            level
                .table
                .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
            level.bucket_live[bucket_idx] += 1;
            level.bucket_summaries[bucket_idx] |= fingerprint_bit(key_fingerprint);
        }
    }

    fn rebuild_special_primary_group(&mut self, group_idx: usize) {
        let mut entries = Vec::new();
        {
            let primary = &mut self.special.primary;
            let group_start = primary.table.group_start(group_idx);
            let group_len = primary.table.group_len(group_idx);
            for slot_idx in group_start..group_start + group_len {
                if primary.table.control_at(slot_idx).is_occupied() {
                    let entry = unsafe { primary.table.take(slot_idx) };
                    entries.push((entry.key, entry.value));
                }
                primary.table.clear_control(slot_idx);
            }
            primary.group_summaries[group_idx] = 0;
            primary.group_tombstones[group_idx] = 0;
        }

        for (key, value) in entries {
            let key_fingerprint = control_fingerprint(self.hash_key(&key));
            let primary = &mut self.special.primary;
            let slot_idx = primary
                .table
                .first_free_in_group(group_idx, 0)
                .expect("rebuilt primary group should have free space");
            primary
                .table
                .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
            primary.group_summaries[group_idx] |= fingerprint_bit(key_fingerprint);
        }
    }

    fn rebuild_special_fallback_bucket(&mut self, bucket_idx: usize) {
        let mut entries = Vec::new();
        {
            let fallback = &mut self.special.fallback;
            let range = fallback.bucket_range(bucket_idx);
            for slot_idx in range.clone() {
                if fallback.table.control_at(slot_idx).is_occupied() {
                    let entry = unsafe { fallback.table.take(slot_idx) };
                    entries.push((entry.key, entry.value));
                }
                fallback.table.clear_control(slot_idx);
            }
            fallback.bucket_live[bucket_idx] = 0;
            fallback.bucket_tombstones[bucket_idx] = 0;
            fallback.bucket_summaries[bucket_idx] = 0;
        }

        for (key, value) in entries {
            let key_fingerprint = control_fingerprint(self.hash_key(&key));
            let fallback = &mut self.special.fallback;
            let range = fallback.bucket_range(bucket_idx);
            let offset = fallback
                .table
                .controls(range.clone())
                .find_first_free()
                .expect("rebuilt fallback bucket should have free space");
            let slot_idx = range.start + offset;
            fallback
                .table
                .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
            fallback.bucket_live[bucket_idx] += 1;
            fallback.bucket_summaries[bucket_idx] |= fingerprint_bit(key_fingerprint);
        }
    }

    fn rebuild_primary_group_summary(&mut self, group_idx: usize) {
        let primary = &mut self.special.primary;
        primary.group_summaries[group_idx] = 0;
        let group_start = primary.table.group_start(group_idx);
        let group_len = primary.table.group_len(group_idx);
        for slot_idx in group_start..group_start + group_len {
            let control = primary.table.control_at(slot_idx);
            if control.is_occupied() {
                primary.group_summaries[group_idx] |= fingerprint_bit(control);
            }
        }
    }

    fn special_fallback_bucket_of_slot(&self, slot_idx: usize) -> usize {
        slot_idx / self.special.fallback.bucket_size.max(1)
    }

    fn shrink_max_populated_level(&mut self) {
        while self.max_populated_level > 0 && self.levels[self.max_populated_level].len == 0 {
            self.max_populated_level -= 1;
        }
        if self.levels.is_empty() || self.levels[0].len == 0 {
            self.max_populated_level = 0;
        }
    }

    fn rebuild_bucket_summary(
        table: &RawTable<Entry<K, V>>,
        range: std::ops::Range<usize>,
        summary: &mut u128,
    ) {
        *summary = 0;
        for slot_idx in range {
            let control = table.control_at(slot_idx);
            if control.is_occupied() {
                *summary |= fingerprint_bit(control);
            }
        }
    }
}

fn compute_level_count(reserve_fraction: f64) -> usize {
    (4.0 * (1.0 / reserve_fraction).log2() + 10.0)
        .ceil()
        .max(1.0) as usize
}

fn compute_bucket_width(reserve_fraction: f64) -> usize {
    (2.0 * (1.0 / reserve_fraction).log2()).ceil().max(1.0) as usize
}

fn log_log_probe_limit(capacity: usize) -> usize {
    let n = capacity.max(4) as f64;
    n.log2().max(2.0).log2().ceil().max(1.0) as usize
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

fn choose_special_capacity(
    total_capacity: usize,
    reserve_fraction: f64,
    bucket_size: usize,
) -> usize {
    if total_capacity == 0 {
        return 0;
    }

    let lower_bound = ((reserve_fraction * total_capacity as f64) / 2.0).ceil() as usize;
    let upper_bound = ((3.0 * reserve_fraction * total_capacity as f64) / 4.0).floor() as usize;
    let lower_bound = lower_bound.min(total_capacity);
    let upper_bound = upper_bound.min(total_capacity);

    if lower_bound <= upper_bound {
        for special_capacity in (lower_bound..=upper_bound).rev() {
            if (total_capacity - special_capacity).is_multiple_of(bucket_size.max(1)) {
                return special_capacity;
            }
        }
    }

    let target = ((5.0 * reserve_fraction * total_capacity as f64) / 8.0)
        .round()
        .clamp(0.0, total_capacity as f64) as usize;

    let mut best_special_capacity = total_capacity % bucket_size.max(1);
    let mut best_distance = usize::MAX;

    for main_capacity in (0..=total_capacity).step_by(bucket_size.max(1)) {
        let special_capacity = total_capacity - main_capacity;
        let distance = special_capacity.abs_diff(target);
        if distance < best_distance {
            best_distance = distance;
            best_special_capacity = special_capacity;
        }
    }

    best_special_capacity
}

fn partition_funnel_buckets(total_buckets: usize, level_count: usize) -> Vec<usize> {
    if level_count == 0 {
        return Vec::new();
    }

    if total_buckets == 0 {
        return vec![0; level_count];
    }

    let first_level_guess = {
        let ratio = 0.75f64;
        let denom = 1.0 - ratio.powi(level_count as i32);
        if denom <= 0.0 {
            total_buckets.max(1)
        } else {
            (((total_buckets as f64) * (1.0 - ratio)) / denom)
                .round()
                .max(0.0) as usize
        }
    };

    for radius in 0..=total_buckets {
        let lower = first_level_guess.saturating_sub(radius);
        if let Some(bucket_counts) = build_funnel_bucket_sequence(total_buckets, level_count, lower)
        {
            return bucket_counts;
        }

        let upper = first_level_guess.saturating_add(radius).min(total_buckets);
        if upper != lower
            && let Some(bucket_counts) =
                build_funnel_bucket_sequence(total_buckets, level_count, upper)
        {
            return bucket_counts;
        }
    }

    let mut fallback_counts = vec![0; level_count];
    fallback_counts[0] = total_buckets;
    fallback_counts
}

fn build_funnel_bucket_sequence(
    total_buckets: usize,
    level_count: usize,
    first_level_bucket_count: usize,
) -> Option<Vec<usize>> {
    if level_count == 0 || first_level_bucket_count > total_buckets {
        return None;
    }

    let mut bucket_counts = Vec::with_capacity(level_count);
    bucket_counts.push(first_level_bucket_count);
    let mut remaining = total_buckets.saturating_sub(first_level_bucket_count);
    let mut previous_bucket_count = first_level_bucket_count;

    for level_idx in 1..level_count {
        let levels_after = level_count - level_idx - 1;
        let (min_next_bucket_count, max_next_bucket_count) =
            next_bucket_count_bounds(previous_bucket_count);
        let ideal_next_bucket_count = ((3 * previous_bucket_count + 2) / 4)
            .clamp(min_next_bucket_count, max_next_bucket_count);

        let chosen_bucket_count = (min_next_bucket_count..=max_next_bucket_count)
            .filter(|&candidate_bucket_count| candidate_bucket_count <= remaining)
            .filter(|&candidate_bucket_count| {
                let remaining_after_candidate = remaining - candidate_bucket_count;
                let (tail_min_sum, tail_max_sum) =
                    possible_tail_sum_range(candidate_bucket_count, levels_after);
                remaining_after_candidate >= tail_min_sum
                    && remaining_after_candidate <= tail_max_sum
            })
            .min_by_key(|&candidate_bucket_count| {
                candidate_bucket_count.abs_diff(ideal_next_bucket_count)
            })?;

        bucket_counts.push(chosen_bucket_count);
        remaining -= chosen_bucket_count;
        previous_bucket_count = chosen_bucket_count;
    }

    if remaining == 0 {
        Some(bucket_counts)
    } else {
        None
    }
}

fn next_bucket_count_bounds(current_bucket_count: usize) -> (usize, usize) {
    let scaled = current_bucket_count.saturating_mul(3);
    let min_next_bucket_count = scaled.saturating_sub(4).div_ceil(4);
    let max_next_bucket_count = scaled.saturating_add(4) / 4;
    (
        min_next_bucket_count,
        max_next_bucket_count.max(min_next_bucket_count),
    )
}

fn possible_tail_sum_range(start_bucket_count: usize, levels_after: usize) -> (usize, usize) {
    let mut min_sum = 0usize;
    let mut max_sum = 0usize;
    let mut min_previous = start_bucket_count;
    let mut max_previous = start_bucket_count;

    for _ in 0..levels_after {
        let (next_min, _) = next_bucket_count_bounds(min_previous);
        let (_, next_max) = next_bucket_count_bounds(max_previous);
        min_sum += next_min;
        max_sum += next_max;
        min_previous = next_min;
        max_previous = next_max;
    }

    (min_sum, max_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn funnel_layout_uses_full_capacity() {
        let capacity = 257;
        let map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(capacity);
        let level_capacity = map.levels.iter().map(BucketLevel::capacity).sum::<usize>();
        let special_capacity =
            map.special.primary.table.capacity() + map.special.fallback.table.capacity();
        assert_eq!(level_capacity + special_capacity, capacity);
    }

    #[test]
    fn insert_get_and_update_work() {
        let mut map = FunnelHashMap::with_capacity(512);

        for key in 0..20 {
            assert_eq!(map.insert(key, key * 10), None);
        }
        for key in 0..20 {
            assert_eq!(map.get(&key), Some(&(key * 10)));
        }

        let replaced = map.insert(7, 777).expect("update should succeed");
        assert_eq!(replaced, 70);
        assert_eq!(map.get(&7), Some(&777));
    }

    #[test]
    fn remove_and_clear_work_with_borrowed_keys() {
        let mut map: FunnelHashMap<String, i32> = FunnelHashMap::with_capacity(256);
        assert_eq!(map.insert("alpha".to_string(), 1), None);
        assert_eq!(map.insert("beta".to_string(), 2), None);

        assert_eq!(map.remove("alpha"), Some(1));
        assert_eq!(map.remove("alpha"), None);
        map.clear();
        assert_eq!(map.get("beta"), None);
        assert!(map.is_empty());
    }

    #[test]
    fn new_starts_empty() {
        let map: FunnelHashMap<i32, i32> = FunnelHashMap::new();
        assert_eq!(map.capacity(), 0);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn insert_resizes_from_zero_capacity() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::new();
        map.insert(1, 10);
        assert_eq!(map.get(&1), Some(&10));
        assert!(map.capacity() > 0);
    }

    #[test]
    fn insert_resizes_when_threshold_is_reached() {
        let capacity = 64;
        let mut map = FunnelHashMap::with_capacity(capacity);
        let max_insertions =
            capacity.saturating_sub((DEFAULT_RESERVE_FRACTION * capacity as f64).floor() as usize);

        for key in 0..max_insertions + 10 {
            let _ = map.insert(key, key * 10);
        }
        for key in 0..max_insertions + 10 {
            assert_eq!(map.get(&key), Some(&(key * 10)));
        }

        assert!(map.capacity() > capacity);
    }

    #[test]
    fn options_constructor_preserves_capacity() {
        let map: FunnelHashMap<i32, i32> = FunnelHashMap::with_options(FunnelOptions {
            capacity: 320,
            reserve_fraction: DEFAULT_RESERVE_FRACTION,
            primary_probe_limit: Some(4),
        });
        assert_eq!(map.capacity(), 320);
    }
}

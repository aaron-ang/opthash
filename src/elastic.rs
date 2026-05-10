use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

use crate::common::DefaultHashBuilder;
use crate::common::simd::ProbeOps;

use crate::common::{
    config::{DEFAULT_RESERVE_FRACTION, INITIAL_CAPACITY},
    control::{CTRL_TOMBSTONE, ControlByte, ControlOps},
    layout::{Entry, GROUP_SIZE, RawTable},
    math::{
        advance_wrapping_index, ceil_three_quarters, fastmod_magic, fastmod_u32,
        floor_half_reserve_slots, level_salt, max_insertions, sanitize_reserve_fraction,
        usize_to_f64,
    },
};

const DEFAULT_PROBE_SCALE: f64 = 16.0;

#[derive(Debug, Clone, Copy)]
pub struct ElasticOptions {
    pub capacity: usize,
    pub reserve_fraction: f64,
    pub probe_scale: f64,
}

impl Default for ElasticOptions {
    fn default() -> Self {
        Self {
            capacity: 0,
            reserve_fraction: DEFAULT_RESERVE_FRACTION,
            probe_scale: DEFAULT_PROBE_SCALE,
        }
    }
}

impl ElasticOptions {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            ..Self::default()
        }
    }
}

#[derive(Debug)]
struct Level<K, V> {
    table: RawTable<Entry<K, V>>,
    len: usize,
    tombstones: usize,
    half_reserve_slot_threshold: usize,
    limited_probe_budgets: Box<[usize]>,
    group_steps: Box<[usize]>,
    salt: u64,
    group_count_magic: u64,
    step_count_magic: u64,
}

impl<K, V> Level<K, V> {
    fn with_capacity(
        capacity: usize,
        reserve_fraction: f64,
        probe_scale: f64,
        level_idx: usize,
    ) -> Self {
        let table = RawTable::new(capacity);
        let group_count = table.group_count();
        let group_steps = ProbeOps::build_group_steps(group_count);
        let limited_probe_budgets =
            build_probe_budgets(capacity, group_count, reserve_fraction, probe_scale);
        let group_count_magic = if group_count > 1 {
            fastmod_magic(group_count)
        } else {
            0
        };
        let step_count_magic = if group_steps.len() > 1 {
            fastmod_magic(group_steps.len())
        } else {
            0
        };

        Self {
            table,
            len: 0,
            tombstones: 0,
            half_reserve_slot_threshold: floor_half_reserve_slots(reserve_fraction, capacity),
            limited_probe_budgets,
            group_steps,
            salt: level_salt(level_idx),
            group_count_magic,
            step_count_magic,
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.table.capacity()
    }

    #[inline]
    fn free_slots(&self) -> usize {
        self.capacity().saturating_sub(self.len)
    }

    #[inline]
    fn limited_group_budget(&self) -> usize {
        self.limited_probe_budgets[self.free_slots()]
    }

    #[inline]
    fn needs_cleanup(&self) -> bool {
        self.tombstones > self.capacity() / 2
    }
}

impl<K, V> Drop for Level<K, V> {
    fn drop(&mut self) {
        for idx in 0..self.table.capacity() {
            if self.table.control_at(idx).is_occupied() {
                unsafe { self.table.drop_in_place(idx) };
            }
        }
    }
}

pub struct ElasticHashMap<K, V> {
    levels: Vec<Level<K, V>>,
    len: usize,
    capacity: usize,
    max_insertions: usize,
    reserve_fraction: f64,
    probe_scale: f64,
    batch_plan: Vec<usize>,
    current_batch_index: usize,
    batch_remaining: usize,
    max_populated_level: usize,
    hash_builder: DefaultHashBuilder,
}

impl<K: std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug for ElasticHashMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElasticHashMap")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("max_populated_level", &self.max_populated_level)
            .finish_non_exhaustive()
    }
}

impl<K, V> Default for ElasticHashMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> ElasticHashMap<K, V>
where
    K: Eq + Hash,
{
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(ElasticOptions::default())
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_options(ElasticOptions::with_capacity(capacity))
    }

    #[must_use]
    pub fn with_options(options: ElasticOptions) -> Self {
        Self::with_options_and_hasher(options, DefaultHashBuilder::default())
    }

    fn with_options_and_hasher(options: ElasticOptions, hash_builder: DefaultHashBuilder) -> Self {
        let reserve_fraction = sanitize_reserve_fraction(options.reserve_fraction);
        let probe_scale = sanitize_probe_scale(options.probe_scale);
        let capacity = options.capacity;
        let max_insertions = max_insertions(capacity, reserve_fraction);

        let level_capacities = partition_levels(capacity);
        let levels = level_capacities
            .iter()
            .enumerate()
            .map(|(level_idx, &cap)| {
                Level::with_capacity(cap, reserve_fraction, probe_scale, level_idx)
            })
            .collect::<Vec<_>>();

        let batch_plan = build_batch_plan(&level_capacities, reserve_fraction, max_insertions);
        let batch_remaining = batch_plan.first().copied().unwrap_or(0);

        Self {
            levels,
            len: 0,
            capacity,
            max_insertions,
            reserve_fraction,
            probe_scale,
            batch_plan,
            current_batch_index: 0,
            batch_remaining,
            max_populated_level: 0,
            hash_builder,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// # Panics
    ///
    /// Panics if a resize succeeds but no free slot can be found for the new key.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let key_hash = self.hash_key(&key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);

        if let Some((level_idx, slot_idx)) =
            self.find_slot_indices_with_hash(&key, key_hash, key_fingerprint)
        {
            let entry = unsafe { self.levels[level_idx].table.get_mut(slot_idx) };
            let old = std::mem::replace(&mut entry.value, value);
            return Some(old);
        }

        if self.len >= self.max_insertions {
            let new_capacity = if self.capacity == 0 {
                INITIAL_CAPACITY
            } else {
                self.capacity.saturating_mul(2)
            };
            self.resize(new_capacity);
        }

        self.advance_batch_window();
        let (level_idx, slot_idx) = self
            .choose_slot_for_new_key(key_hash)
            .expect("no free slot found after resize");

        let level = &mut self.levels[level_idx];
        let prev_ctrl = level.table.control_at(slot_idx);
        level
            .table
            .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
        level.len += 1;
        if prev_ctrl == CTRL_TOMBSTONE {
            level.tombstones -= 1;
        }
        if level_idx > self.max_populated_level {
            self.max_populated_level = level_idx;
        }
        self.len += 1;
        if self.batch_remaining > 0 {
            self.batch_remaining -= 1;
        }
        None
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);
        let (level_idx, slot_idx) =
            self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)?;
        Some(unsafe { &self.levels[level_idx].table.get_ref(slot_idx).value })
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);
        let (level_idx, slot_idx) =
            self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)?;
        Some(unsafe { &mut self.levels[level_idx].table.get_mut(slot_idx).value })
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);
        self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)
            .is_some()
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);
        let (level_idx, slot_idx) =
            self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)?;

        let removed_entry = {
            let level = &mut self.levels[level_idx];
            let removed = unsafe { level.table.take(slot_idx) };
            level.table.mark_tombstone(slot_idx);
            level.len -= 1;
            level.tombstones += 1;
            removed
        };

        self.len -= 1;
        let needs_resize = self.levels[level_idx].needs_cleanup();
        self.shrink_max_populated_level();
        if needs_resize {
            self.resize(self.capacity);
        }
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
            level.tombstones = 0;
        }
        self.len = 0;
        self.current_batch_index = 0;
        self.batch_remaining = self.batch_plan.first().copied().unwrap_or(0);
        self.max_populated_level = 0;
    }

    #[must_use]
    pub fn iter(&self) -> ElasticIter<'_, K, V> {
        ElasticIter {
            levels: &self.levels,
            level_idx: 0,
            slot_idx: 0,
        }
    }
}

pub struct ElasticIter<'a, K, V> {
    levels: &'a [Level<K, V>],
    level_idx: usize,
    slot_idx: usize,
}

impl<'a, K, V> Iterator for ElasticIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.level_idx < self.levels.len() {
            let level = &self.levels[self.level_idx];
            while self.slot_idx < level.table.capacity() {
                let idx = self.slot_idx;
                self.slot_idx += 1;
                if level.table.control_at(idx).is_occupied() {
                    let entry = unsafe { level.table.get_ref(idx) };
                    return Some((&entry.key, &entry.value));
                }
            }
            self.level_idx += 1;
            self.slot_idx = 0;
        }
        None
    }
}

impl<'a, K, V> IntoIterator for &'a ElasticHashMap<K, V>
where
    K: Eq + Hash,
{
    type Item = (&'a K, &'a V);
    type IntoIter = ElasticIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> ElasticHashMap<K, V>
where
    K: Eq + Hash,
{
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
            level.tombstones = 0;
        }

        self.len = 0;
        self.max_populated_level = 0;

        let hash_builder = std::mem::take(&mut self.hash_builder);
        let mut new_map = Self::with_options_and_hasher(
            ElasticOptions {
                capacity: new_capacity,
                reserve_fraction: self.reserve_fraction,
                probe_scale: self.probe_scale,
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
    fn advance_batch_window(&mut self) {
        while self.batch_remaining == 0 && self.current_batch_index + 1 < self.batch_plan.len() {
            self.current_batch_index += 1;
            self.batch_remaining = self.batch_plan[self.current_batch_index];
        }
    }

    fn choose_slot_for_new_key(&mut self, key_hash: u64) -> Option<(usize, usize)> {
        if self.levels.is_empty() {
            return None;
        }

        if let Some(pair) = self.choose_slot_targeted(key_hash) {
            return Some(pair);
        }

        for li in 0..self.levels.len() {
            if let Some(slot_idx) = self.first_free_uniform(key_hash, li) {
                return Some((li, slot_idx));
            }
        }
        None
    }

    fn choose_slot_targeted(&self, key_hash: u64) -> Option<(usize, usize)> {
        if self.current_batch_index == 0 {
            return self
                .first_free_uniform(key_hash, 0)
                .map(|slot_idx| (0, slot_idx));
        }

        let level_idx = self.current_batch_index.saturating_sub(1);
        if level_idx + 1 >= self.levels.len() {
            let last = self.levels.len() - 1;
            return self
                .first_free_uniform(key_hash, last)
                .map(|slot_idx| (last, slot_idx));
        }

        let current_level = &self.levels[level_idx];
        let next_level = &self.levels[level_idx + 1];
        let current_free_slots = current_level.free_slots();
        let next_free_slots = next_level.free_slots();

        if current_free_slots > current_level.half_reserve_slot_threshold
            && next_free_slots.saturating_mul(4) > next_level.capacity()
        {
            let limited_budget = current_level.limited_group_budget();
            if let Some(slot_idx) = self.first_free_limited(key_hash, level_idx, limited_budget) {
                return Some((level_idx, slot_idx));
            }
            if let Some(slot_idx) = self.first_free_uniform(key_hash, level_idx + 1) {
                return Some((level_idx + 1, slot_idx));
            }
            return self
                .first_free_uniform(key_hash, level_idx)
                .map(|slot_idx| (level_idx, slot_idx));
        }

        if current_free_slots <= current_level.half_reserve_slot_threshold {
            if let Some(slot_idx) = self.first_free_uniform(key_hash, level_idx + 1) {
                return Some((level_idx + 1, slot_idx));
            }
            return self
                .first_free_uniform(key_hash, level_idx)
                .map(|slot_idx| (level_idx, slot_idx));
        }

        if let Some(slot_idx) = self.first_free_uniform(key_hash, level_idx) {
            return Some((level_idx, slot_idx));
        }
        self.first_free_uniform(key_hash, level_idx + 1)
            .map(|slot_idx| (level_idx + 1, slot_idx))
    }

    fn find_slot_indices_with_hash<Q>(
        &self,
        key: &Q,
        key_hash: u64,
        key_fingerprint: u8,
    ) -> Option<(usize, usize)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let search_limit = (self.max_populated_level + 1).min(self.levels.len());
        for (level_idx, level) in self.levels[..search_limit].iter().enumerate() {
            if let Some(slot_idx) =
                Self::find_in_level_by_probe(key_hash, key_fingerprint, key, level)
            {
                return Some((level_idx, slot_idx));
            }
        }
        None
    }

    #[inline]
    fn find_in_level_by_probe<Q>(
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
        level: &Level<K, V>,
    ) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if level.capacity() == 0 || level.len == 0 {
            return None;
        }

        let (group_start, group_step) = Self::group_probe_params(level, key_hash);
        let group_count = level.table.group_count();
        let mut group_idx = group_start;

        let tombstone_free = level.tombstones == 0;
        let capacity = level.capacity();
        for _ in 0..group_count {
            let (match_mask, free_mask) = level
                .table
                .group_match_and_free_mask(group_idx, key_fingerprint);
            for relative_idx in match_mask {
                let slot_idx = group_idx * GROUP_SIZE + relative_idx;
                let entry = unsafe { level.table.get_ref(slot_idx) };
                if entry.key.borrow() == key {
                    return Some(slot_idx);
                }
            }

            if tombstone_free
                && let Some(off) = free_mask.lowest()
                && group_idx * GROUP_SIZE + off < capacity
            {
                return None;
            }

            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }

        None
    }

    fn first_free_limited(
        &self,
        key_hash: u64,
        level_idx: usize,
        max_groups: usize,
    ) -> Option<usize> {
        let level = &self.levels[level_idx];
        if level.capacity() == 0 || level.len >= level.capacity() {
            return None;
        }

        let (group_start, group_step) = Self::group_probe_params(level, key_hash);
        let group_count = level.table.group_count();
        let mut group_idx = group_start;
        let max_groups = max_groups.min(group_count.max(1));

        for _ in 0..max_groups {
            if let Some(slot_idx) = level.table.first_free_in_group(group_idx) {
                return Some(slot_idx);
            }
            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }

        None
    }

    fn first_free_uniform(&self, key_hash: u64, level_idx: usize) -> Option<usize> {
        let level = &self.levels[level_idx];
        if level.capacity() == 0 || level.len >= level.capacity() {
            return None;
        }

        let (group_start, group_step) = Self::group_probe_params(level, key_hash);
        let group_count = level.table.group_count();
        let mut group_idx = group_start;

        for _ in 0..group_count {
            if let Some(slot_idx) = level.table.first_free_in_group(group_idx) {
                return Some(slot_idx);
            }
            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }

        None
    }

    #[inline]
    fn group_probe_params(level: &Level<K, V>, key_hash: u64) -> (usize, usize) {
        let group_count = level.table.group_count();
        if group_count <= 1 {
            return (0, 1);
        }

        let mixed = key_hash ^ level.salt;
        let group_start = fastmod_u32(mixed, level.group_count_magic, group_count);
        let step = if level.group_steps.len() > 1 {
            let step_idx = fastmod_u32(
                mixed.rotate_left(29),
                level.step_count_magic,
                level.group_steps.len(),
            );
            level.group_steps[step_idx]
        } else {
            level.group_steps[0]
        };
        (group_start, step)
    }

    fn shrink_max_populated_level(&mut self) {
        while self.max_populated_level > 0
            && self.levels[self.max_populated_level].len == 0
            && self.levels[self.max_populated_level].tombstones == 0
        {
            self.max_populated_level -= 1;
        }
        if self.levels.is_empty() || (self.levels[0].len == 0 && self.levels[0].tombstones == 0) {
            self.max_populated_level = 0;
        }
    }
}

fn sanitize_probe_scale(probe_scale: f64) -> f64 {
    if probe_scale.is_finite() && probe_scale > 0.0 {
        probe_scale
    } else {
        DEFAULT_PROBE_SCALE
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn build_probe_budgets(
    capacity: usize,
    group_count: usize,
    reserve_fraction: f64,
    probe_scale: f64,
) -> Box<[usize]> {
    let mut budgets = vec![1usize; capacity.saturating_add(1)];
    if capacity == 0 {
        return budgets.into_boxed_slice();
    }

    let max_budget = group_count.max(1);
    let cap_f = usize_to_f64(capacity);
    let log_cap = (1.0 / reserve_fraction).log2();

    // Budget(fs) is a non-increasing staircase function of free_slots.
    // Instead of computing log2/ceil per slot, find the threshold free_slots
    // where each budget level transitions, then fill segments.
    //
    // Budget >= b when: fs < capacity / 2^sqrt((b-1)*GROUP_SIZE / probe_scale)
    let mut thresholds: Vec<(usize, usize)> = Vec::new();
    for b in 2..=max_budget {
        let ratio = ((b - 1) * GROUP_SIZE) as f64 / probe_scale;
        if ratio >= log_cap {
            break;
        }
        let exact = cap_f / f64::exp2(ratio.sqrt());
        let threshold = (exact.ceil() as usize).saturating_sub(1).min(capacity);
        if threshold == 0 {
            break;
        }
        thresholds.push((b, threshold));
    }

    // Fill from highest budget inward (thresholds decrease with increasing b).
    let mut prev_end = 0;
    for &(b, threshold) in thresholds.iter().rev() {
        if threshold > prev_end {
            budgets[(prev_end + 1)..=threshold].fill(b);
            prev_end = threshold;
        }
    }

    budgets.into_boxed_slice()
}

fn partition_levels(total_capacity: usize) -> Vec<usize> {
    if total_capacity == 0 {
        return Vec::new();
    }

    let mut sizes = Vec::new();
    let mut remaining = total_capacity;
    let mut next_size = total_capacity.div_ceil(2);

    while remaining > 0 {
        let size = next_size.min(remaining).max(1);
        sizes.push(size);
        remaining -= size;
        if remaining == 0 {
            break;
        }
        next_size = (size / 2).max(1);
    }

    sizes
}

fn build_batch_plan(
    level_capacities: &[usize],
    reserve_fraction: f64,
    max_insertions: usize,
) -> Vec<usize> {
    if level_capacities.is_empty() || max_insertions == 0 {
        return Vec::new();
    }

    let mut plan = Vec::with_capacity(level_capacities.len() + 1);
    plan.push(ceil_three_quarters(level_capacities[0]));

    for level_index in 1..level_capacities.len() {
        let current_level_capacity = level_capacities[level_index - 1];
        let next_level_capacity = level_capacities[level_index];

        let target_current_level_occupancy = current_level_capacity.saturating_sub(
            floor_half_reserve_slots(reserve_fraction, current_level_capacity),
        );
        let initial_current_level_occupancy = ceil_three_quarters(current_level_capacity);
        let initial_next_level_occupancy = ceil_three_quarters(next_level_capacity);

        let batch_size = target_current_level_occupancy
            .saturating_sub(initial_current_level_occupancy)
            .saturating_add(initial_next_level_occupancy);
        plan.push(batch_size);
    }

    let mut inserted = 0;
    for size in &mut plan {
        if inserted >= max_insertions {
            *size = 0;
            continue;
        }
        let room = max_insertions - inserted;
        if *size > room {
            *size = room;
        }
        inserted += *size;
    }

    if inserted < max_insertions {
        plan.push(max_insertions - inserted);
    }

    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_partition_keeps_capacity_and_halving_shape() {
        let sizes = partition_levels(127);
        assert_eq!(sizes.iter().sum::<usize>(), 127);
        assert!(!sizes.is_empty());

        for window in sizes.windows(2) {
            let current_level_size = usize_to_f64(window[0]);
            let next_level_size = usize_to_f64(window[1]);
            assert!(next_level_size >= (current_level_size / 2.0 - 1.0));
            assert!(next_level_size <= (current_level_size / 2.0 + 1.0));
        }
    }

    #[test]
    fn insert_get_and_update_work() {
        let mut map = ElasticHashMap::with_capacity(64);

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
    fn get_mut_and_contains_key_work() {
        let mut map = ElasticHashMap::new();
        assert_eq!(map.insert("alpha", 1), None);
        assert!(map.contains_key("alpha"));

        if let Some(v) = map.get_mut("alpha") {
            *v = 2;
        }
        assert_eq!(map.get("alpha"), Some(&2));
    }

    #[test]
    fn remove_supports_borrowed_key_and_updates_len() {
        let mut map: ElasticHashMap<String, i32> = ElasticHashMap::new();
        assert_eq!(map.insert("alpha".to_string(), 1), None);
        assert_eq!(map.insert("beta".to_string(), 2), None);

        assert_eq!(map.remove("alpha"), Some(1));
        assert_eq!(map.remove("alpha"), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("beta"), Some(&2));
    }

    #[test]
    fn clear_removes_all_entries_and_resets_map() {
        let mut map = ElasticHashMap::with_capacity(64);
        for key in 0..10 {
            assert_eq!(map.insert(key, key * 10), None);
        }

        map.clear();
        assert!(map.is_empty());
        for key in 0..10 {
            assert_eq!(map.get(&key), None);
        }

        assert_eq!(map.insert(99, 990), None);
        assert_eq!(map.get(&99), Some(&990));
    }

    #[test]
    fn new_starts_with_zero_capacity() {
        let map: ElasticHashMap<i32, i32> = ElasticHashMap::new();
        assert_eq!(map.capacity(), 0);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn insert_resizes_when_threshold_is_reached() {
        let capacity = 40;
        let mut map = ElasticHashMap::with_capacity(capacity);
        let max_insertions = max_insertions(capacity, DEFAULT_RESERVE_FRACTION);

        for key in 0..max_insertions + 10 {
            assert_eq!(map.insert(key, key), None);
        }

        for key in 0..max_insertions + 10 {
            assert_eq!(map.get(&key), Some(&key));
        }

        assert!(map.capacity() > capacity);
    }

    #[test]
    fn insert_resizes_from_zero_capacity() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::new();
        map.insert(1, 10);
        assert_eq!(map.get(&1), Some(&10));
        assert!(map.capacity() > 0);
    }

    #[test]
    fn options_constructor_preserves_capacity() {
        let map: ElasticHashMap<i32, i32> = ElasticHashMap::with_options(ElasticOptions {
            capacity: 96,
            reserve_fraction: DEFAULT_RESERVE_FRACTION,
            probe_scale: 8.0,
        });
        assert_eq!(map.capacity(), 96);
    }

    #[test]
    fn delete_heavy_preserves_correctness() {
        let n = 10_000;
        let cutoff = (n * 4) / 5;
        for trial in 0..50 {
            let mut map = ElasticHashMap::new();
            for i in 0..n {
                map.insert(i, i * 10);
            }
            // Delete the first 80% of keys.
            for i in 0..cutoff {
                assert_eq!(
                    map.remove(&i),
                    Some(i * 10),
                    "trial {trial}: missing key {i} during delete"
                );
            }
            // Lookup remaining keys (post-tombstone state).
            for i in cutoff..n {
                assert_eq!(
                    map.get(&i),
                    Some(&(i * 10)),
                    "trial {trial}: key {i} missing after deletes"
                );
            }
            assert_eq!(map.len(), (n - cutoff) as usize);
            // Re-insert into tombstone-heavy map.
            for i in n..(n + n / 5) {
                assert_eq!(map.insert(i, i), None);
            }
            for i in n..(n + n / 5) {
                assert_eq!(
                    map.get(&i),
                    Some(&i),
                    "trial {trial}: key {i} missing after re-insert"
                );
            }
        }
    }

    #[test]
    fn large_map_correctness() {
        let n = 10_000;
        let mut map = ElasticHashMap::with_capacity(n * 2);
        for i in 0..n {
            assert_eq!(map.insert(i, i), None);
        }
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&i), "key {i} missing");
        }
        assert_eq!(map.len(), n);
    }

    #[test]
    fn partial_group_capacity_works() {
        // Capacity 18 creates a partial last group (2 valid slots out of 16).
        let mut map = ElasticHashMap::with_capacity(18);
        for i in 0..15 {
            assert_eq!(map.insert(i, i), None);
        }
        for i in 0..15 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn iter_yields_every_inserted_pair_once() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(64);
        for i in 0..50 {
            map.insert(i, i * 10);
        }
        let mut collected: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        collected.sort();
        let expected: Vec<(i32, i32)> = (0..50).map(|i| (i, i * 10)).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn iter_skips_tombstones_after_remove() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(32);
        for i in 0..20 {
            map.insert(i, i);
        }
        for i in (0..20).step_by(2) {
            map.remove(&i);
        }
        let keys: Vec<i32> = map.iter().map(|(&k, _)| k).collect();
        assert_eq!(keys.len(), 10);
        let mut sorted = keys;
        sorted.sort();
        assert_eq!(sorted, (1..20).step_by(2).collect::<Vec<_>>());
    }

    #[test]
    fn iter_empty_map_is_empty() {
        let map: ElasticHashMap<i32, i32> = ElasticHashMap::new();
        assert_eq!(map.iter().count(), 0);
    }
}

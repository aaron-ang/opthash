use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

use crate::common::{
    CTRL_EMPTY, CTRL_TOMBSTONE, DEFAULT_RESERVE_FRACTION, Entry, RawSlots,
    advance_wrapping_index, ceil_three_quarters, control_fingerprint, find_first_free_control,
    floor_half_reserve_slots, greatest_common_divisor, is_free_control, is_occupied_control,
    sanitize_reserve_fraction,
};

const DEFAULT_PROBE_SCALE: f64 = 16.0;
const INITIAL_CAPACITY: usize = 16;

#[derive(Debug)]
struct Level<K, V> {
    slots: RawSlots<Entry<K, V>>,
    controls: Vec<u8>,
    len: usize,
}

impl<K, V> Level<K, V> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: RawSlots::new(capacity),
            controls: vec![CTRL_EMPTY; capacity],
            len: 0,
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn free_fraction(&self) -> f64 {
        if self.capacity() == 0 {
            0.0
        } else {
            1.0 - (self.len as f64 / self.capacity() as f64)
        }
    }
}

impl<K, V> Drop for Level<K, V> {
    fn drop(&mut self) {
        for (idx, &control) in self.controls.iter().enumerate() {
            if is_occupied_control(control) {
                // SAFETY: occupied control means the slot was initialized via write.
                unsafe { self.slots.drop_in_place(idx) };
            }
        }
    }
}

#[derive(Debug)]
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
    hash_builder: RandomState,
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
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_params(capacity, DEFAULT_RESERVE_FRACTION, DEFAULT_PROBE_SCALE)
    }

    fn with_params(capacity: usize, reserve_fraction: f64, probe_scale: f64) -> Self {
        Self::with_params_and_hasher(capacity, reserve_fraction, probe_scale, RandomState::new())
    }

    fn with_params_and_hasher(
        capacity: usize,
        reserve_fraction: f64,
        probe_scale: f64,
        hash_builder: RandomState,
    ) -> Self {
        let reserve_fraction = sanitize_reserve_fraction(reserve_fraction);
        let probe_scale = sanitize_probe_scale(probe_scale);

        let level_capacities = partition_levels(capacity);
        let levels = level_capacities
            .iter()
            .map(|&cap| Level::with_capacity(cap))
            .collect::<Vec<_>>();

        let max_insertions =
            capacity.saturating_sub((reserve_fraction * capacity as f64).floor() as usize);
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

        if let Some((level_idx, slot_idx)) =
            self.find_slot_indices_with_hash(&key, key_hash, key_fingerprint)
        {
            // SAFETY: find_slot_indices_with_hash only returns occupied slots.
            let entry = unsafe { self.levels[level_idx].slots.get_mut(slot_idx) };
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
            .unwrap_or_else(|| {
                panic!(
                    "ElasticHashMap: no free slot found; resize not yet implemented (capacity={}, len={})",
                    self.capacity, self.len
                )
            });

        self.levels[level_idx]
            .slots
            .write(slot_idx, Entry { key, value });
        self.levels[level_idx].controls[slot_idx] = key_fingerprint;
        self.levels[level_idx].len += 1;
        self.len += 1;
        if level_idx > self.max_populated_level {
            self.max_populated_level = level_idx;
        }
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
        let key_fingerprint = control_fingerprint(key_hash);
        let (level_idx, slot_idx) =
            self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)?;
        // SAFETY: find_slot_indices_with_hash only returns occupied slots.
        Some(unsafe { &self.levels[level_idx].slots.get_ref(slot_idx).value })
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = control_fingerprint(key_hash);
        let (level_idx, slot_idx) =
            self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)?;
        // SAFETY: find_slot_indices_with_hash only returns occupied slots.
        Some(unsafe { &mut self.levels[level_idx].slots.get_mut(slot_idx).value })
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = control_fingerprint(key_hash);
        self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)
            .is_some()
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = control_fingerprint(key_hash);
        let (level_idx, slot_idx) =
            self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)?;
        let level = &mut self.levels[level_idx];
        // SAFETY: find_slot_indices_with_hash only returns occupied slots.
        let removed_entry = unsafe { level.slots.take(slot_idx) };
        level.controls[slot_idx] = CTRL_TOMBSTONE;
        level.len -= 1;
        self.len -= 1;
        Some(removed_entry.value)
    }

    pub fn clear(&mut self) {
        for level in &mut self.levels {
            for (idx, control) in level.controls.iter_mut().enumerate() {
                if is_occupied_control(*control) {
                    // SAFETY: occupied control means slot was initialized.
                    unsafe { level.slots.drop_in_place(idx) };
                }
                *control = CTRL_EMPTY;
            }
            level.len = 0;
        }
        self.len = 0;
        self.current_batch_index = 0;
        self.batch_remaining = self.batch_plan.first().copied().unwrap_or(0);
        self.max_populated_level = 0;
    }

    fn resize(&mut self, new_capacity: usize) {
        let mut entries = Vec::with_capacity(self.len);

        for level in &mut self.levels {
            for (idx, control) in level.controls.iter_mut().enumerate() {
                if is_occupied_control(*control) {
                    let entry = unsafe { level.slots.take(idx) };
                    entries.push((entry.key, entry.value));
                    *control = CTRL_EMPTY;
                }
            }
            level.len = 0;
        }
        self.len = 0;
        self.max_populated_level = 0;

        let hash_builder =
            std::mem::replace(&mut self.hash_builder, RandomState::new());
        let mut new_map = Self::with_params_and_hasher(
            new_capacity,
            self.reserve_fraction,
            self.probe_scale,
            hash_builder,
        );

        for (key, value) in entries {
            new_map.insert(key, value);
        }

        *self = new_map;
    }

    #[inline]
    fn advance_batch_window(&mut self) {
        while self.batch_remaining == 0 && self.current_batch_index + 1 < self.batch_plan.len() {
            self.current_batch_index += 1;
            self.batch_remaining = self.batch_plan[self.current_batch_index];
        }
    }

    #[inline]
    fn choose_slot_for_new_key(&self, key_hash: u64) -> Option<(usize, usize)> {
        if self.levels.is_empty() {
            return None;
        }

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

        let curr_level_free_fraction = self.levels[level_idx].free_fraction();
        let next_level_free_fraction = self.levels[level_idx + 1].free_fraction();

        if curr_level_free_fraction > self.reserve_fraction / 2.0 && next_level_free_fraction > 0.25
        {
            let probe_limit = self.probe_limit_for_free_fraction(curr_level_free_fraction);
            if let Some(slot_idx) = self.first_free_limited(key_hash, level_idx, probe_limit) {
                return Some((level_idx, slot_idx));
            }
            if let Some(slot_idx) = self.first_free_uniform(key_hash, level_idx + 1) {
                return Some((level_idx + 1, slot_idx));
            }
            return self
                .first_free_uniform(key_hash, level_idx)
                .map(|slot_idx| (level_idx, slot_idx));
        }

        if curr_level_free_fraction <= self.reserve_fraction / 2.0 {
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

    #[inline]
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
        for level_idx in 0..search_limit {
            if let Some(slot_idx) =
                self.find_in_level_by_probe(key_hash, key_fingerprint, key, level_idx)
            {
                return Some((level_idx, slot_idx));
            }
        }
        None
    }

    #[inline]
    fn probe_limit_for_free_fraction(&self, free_fraction: f64) -> usize {
        let bounded_free_fraction = free_fraction.clamp(1e-12, 1.0);
        let log_sq_inverse_free_fraction = (1.0 / bounded_free_fraction).log2().powi(2);
        let log_inverse_reserve_fraction = (1.0 / self.reserve_fraction).log2();
        let probe_budget =
            self.probe_scale * log_sq_inverse_free_fraction.min(log_inverse_reserve_fraction);
        probe_budget.ceil().max(1.0) as usize
    }

    #[inline]
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    #[inline]
    fn find_in_level_by_probe<Q>(
        &self,
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
        level_idx: usize,
    ) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let level = &self.levels[level_idx];
        let level_cap = level.capacity();
        if level_cap == 0 || level.len == 0 {
            return None;
        }

        let (probe_start, probe_step) =
            self.probe_sequence_start_step(key_hash, level_idx, level_cap);
        let mut slot_idx = probe_start;
        for _ in 0..level_cap {
            let control = level.controls[slot_idx];
            if control == CTRL_EMPTY {
                return None;
            }
            if control == key_fingerprint {
                // SAFETY: fingerprint match means slot is occupied (fingerprints are 1..=0x7F).
                let entry = unsafe { level.slots.get_ref(slot_idx) };
                if entry.key.borrow() == key {
                    return Some(slot_idx);
                }
            }
            slot_idx = advance_wrapping_index(slot_idx, probe_step, level_cap);
        }

        None
    }

    #[inline]
    fn probe_sequence_start_step(
        &self,
        key_hash: u64,
        level_idx: usize,
        level_cap: usize,
    ) -> (usize, usize) {
        let rotated = key_hash.rotate_left((level_idx as u32 * 7) % 64);
        let probe_start = (rotated as usize) % level_cap;
        let mut probe_step = (rotated.rotate_left(32) as usize) % level_cap;
        if probe_step == 0 {
            probe_step = 1;
        }

        while greatest_common_divisor(probe_step, level_cap) != 1 {
            probe_step += 1;
            if probe_step == level_cap {
                probe_step = 1;
            }
        }

        (probe_start, probe_step)
    }

    #[inline]
    fn first_free_limited(
        &self,
        key_hash: u64,
        level_idx: usize,
        probe_limit: usize,
    ) -> Option<usize> {
        let level = &self.levels[level_idx];
        let level_cap = level.capacity();
        if level_cap == 0 || level.len >= level_cap {
            return None;
        }

        let (probe_start, probe_step) =
            self.probe_sequence_start_step(key_hash, level_idx, level_cap);
        let max_probes = probe_limit.min(level_cap);
        if probe_step == 1 {
            let first_run = max_probes.min(level_cap - probe_start);
            if let Some(offset) =
                find_first_free_control(&level.controls[probe_start..probe_start + first_run])
            {
                return Some(probe_start + offset);
            }

            let wrapped = max_probes - first_run;
            if wrapped > 0
                && let Some(offset) = find_first_free_control(&level.controls[..wrapped])
            {
                return Some(offset);
            }
            return None;
        }

        let mut slot_idx = probe_start;
        for _ in 0..max_probes {
            if is_free_control(level.controls[slot_idx]) {
                return Some(slot_idx);
            }
            slot_idx = advance_wrapping_index(slot_idx, probe_step, level_cap);
        }
        None
    }

    #[inline]
    fn first_free_uniform(&self, key_hash: u64, level_idx: usize) -> Option<usize> {
        let level = &self.levels[level_idx];
        let level_cap = level.capacity();
        if level_cap == 0 || level.len >= level_cap {
            return None;
        }

        let (probe_start, probe_step) =
            self.probe_sequence_start_step(key_hash, level_idx, level_cap);
        if probe_step == 1 {
            if let Some(offset) = find_first_free_control(&level.controls[probe_start..]) {
                return Some(probe_start + offset);
            }
            return find_first_free_control(&level.controls[..probe_start]);
        }

        let mut slot_idx = probe_start;
        for _ in 0..level_cap {
            if is_free_control(level.controls[slot_idx]) {
                return Some(slot_idx);
            }
            slot_idx = advance_wrapping_index(slot_idx, probe_step, level_cap);
        }

        None
    }
}

fn sanitize_probe_scale(probe_scale: f64) -> f64 {
    if probe_scale.is_finite() && probe_scale > 0.0 {
        probe_scale
    } else {
        DEFAULT_PROBE_SCALE
    }
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
            let current_level_size = window[0] as f64;
            let next_level_size = window[1] as f64;
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
        let max_insertions =
            capacity.saturating_sub((DEFAULT_RESERVE_FRACTION * capacity as f64).floor() as usize);

        // Insert beyond the original threshold
        for key in 0..max_insertions + 10 {
            assert_eq!(map.insert(key, key), None);
        }

        // Verify all entries are still accessible
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
}

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

use crate::common::{
    DEFAULT_CAPACITY, DEFAULT_RESERVE_FRACTION, Entry, ceil_three_quarters, empty_slots,
    floor_half_reserve_slots, sanitize_reserve_fraction,
};

const DEFAULT_PROBE_SCALE: f64 = 16.0;
const DOMAIN_PROBE_START: u8 = 1;
const DOMAIN_PROBE_STEP: u8 = 2;

#[derive(Debug)]
struct Level<K, V> {
    slots: Vec<Option<Entry<K, V>>>,
    len: usize,
}

impl<K, V> Level<K, V> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: empty_slots(capacity),
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn free_fraction(&self) -> f64 {
        if self.capacity() == 0 {
            0.0
        } else {
            1.0 - (self.len as f64 / self.capacity() as f64)
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
        Self::with_params(
            DEFAULT_CAPACITY,
            DEFAULT_RESERVE_FRACTION,
            DEFAULT_PROBE_SCALE,
        )
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_params(capacity, DEFAULT_RESERVE_FRACTION, DEFAULT_PROBE_SCALE)
    }

    fn with_params(capacity: usize, reserve_fraction: f64, probe_scale: f64) -> Self {
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
            hash_builder: RandomState::new(),
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
        if let Some((level_idx, slot_idx)) = self.find_slot_indices(&key) {
            let entry = self.levels[level_idx].slots[slot_idx]
                .as_mut()
                .expect("slot index returned by find_slot_indices must be occupied");
            let old = std::mem::replace(&mut entry.value, value);
            return Some(old);
        }

        if self.len >= self.max_insertions {
            todo!("resize on capacity reached");
        }

        self.advance_batch_window();
        let (level_idx, slot_idx) = self
            .choose_slot_for_new_key(&key)
            .unwrap_or_else(|| todo!("resize on capacity reached"));

        self.levels[level_idx].slots[slot_idx] = Some(Entry { key, value });
        self.levels[level_idx].len += 1;
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
        if let Some((level_idx, slot_idx)) = self.find_slot_indices(key) {
            return self.levels[level_idx].slots[slot_idx]
                .as_ref()
                .map(|entry| &entry.value);
        }
        None
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (level_idx, slot_idx) = self.find_slot_indices(key)?;
        self.levels[level_idx].slots[slot_idx]
            .as_mut()
            .map(|entry| &mut entry.value)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find_slot_indices(key).is_some()
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (level_idx, slot_idx) = self.find_slot_indices(key)?;
        let level = &mut self.levels[level_idx];
        let removed_entry = level.slots[slot_idx].take()?;
        level.len -= 1;
        self.len -= 1;
        Some(removed_entry.value)
    }

    pub fn clear(&mut self) {
        for level in &mut self.levels {
            for slot in &mut level.slots {
                *slot = None;
            }
            level.len = 0;
        }
        self.len = 0;
        self.current_batch_index = 0;
        self.batch_remaining = self.batch_plan.first().copied().unwrap_or(0);
    }

    fn advance_batch_window(&mut self) {
        while self.batch_remaining == 0 && self.current_batch_index + 1 < self.batch_plan.len() {
            self.current_batch_index += 1;
            self.batch_remaining = self.batch_plan[self.current_batch_index];
        }
    }

    fn choose_slot_for_new_key(&self, key: &K) -> Option<(usize, usize)> {
        if self.levels.is_empty() {
            return None;
        }

        let key_hash = self.hash_key(key);

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

    fn find_slot_indices<Q>(&self, key: &Q) -> Option<(usize, usize)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        for level_idx in 0..self.levels.len() {
            if let Some(slot_idx) = self.find_in_level_by_probe(key_hash, key, level_idx) {
                return Some((level_idx, slot_idx));
            }
        }
        None
    }

    fn probe_limit_for_free_fraction(&self, free_fraction: f64) -> usize {
        let bounded_free_fraction = free_fraction.clamp(1e-12, 1.0);
        let log_inverse_free_fraction = (1.0 / bounded_free_fraction).log2();
        let log_inverse_reserve_fraction = (1.0 / self.reserve_fraction).log2();
        let probe_budget =
            self.probe_scale * log_inverse_free_fraction.min(log_inverse_reserve_fraction);
        probe_budget.ceil().max(1.0) as usize
    }

    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    fn find_in_level_by_probe<Q>(&self, key_hash: u64, key: &Q, level_idx: usize) -> Option<usize>
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
        for probe in 0..level_cap {
            let slot_idx = probe_slot_index(probe_start, probe_step, probe, level_cap);
            if level.slots[slot_idx]
                .as_ref()
                .is_some_and(|entry| entry.key.borrow() == key)
            {
                return Some(slot_idx);
            }
        }

        None
    }

    fn probe_sequence_start_step(
        &self,
        key_hash: u64,
        level_idx: usize,
        level_cap: usize,
    ) -> (usize, usize) {
        let probe_start = (self
            .hash_builder
            .hash_one((DOMAIN_PROBE_START, key_hash, level_idx))
            as usize)
            % level_cap;
        let mut probe_step = (self
            .hash_builder
            .hash_one((DOMAIN_PROBE_STEP, key_hash, level_idx))
            as usize)
            % level_cap;
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
        for probe in 0..probe_limit.min(level_cap) {
            let slot_idx = probe_slot_index(probe_start, probe_step, probe, level_cap);
            if level.slots[slot_idx].is_none() {
                return Some(slot_idx);
            }
        }
        None
    }

    fn first_free_uniform(&self, key_hash: u64, level_idx: usize) -> Option<usize> {
        let level = &self.levels[level_idx];
        let level_cap = level.capacity();
        if level_cap == 0 || level.len >= level_cap {
            return None;
        }

        let (probe_start, probe_step) =
            self.probe_sequence_start_step(key_hash, level_idx, level_cap);
        for probe in 0..level_cap {
            let slot_idx = probe_slot_index(probe_start, probe_step, probe, level_cap);
            if level.slots[slot_idx].is_none() {
                return Some(slot_idx);
            }
        }

        None
    }
}

fn probe_slot_index(
    probe_start: usize,
    probe_step: usize,
    probe: usize,
    level_cap: usize,
) -> usize {
    (probe_start + probe.saturating_mul(probe_step)) % level_cap
}

fn greatest_common_divisor(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
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
    fn insert_panics_with_resize_todo_when_threshold_is_reached() {
        let capacity = 40;
        let mut map = ElasticHashMap::with_capacity(capacity);
        let max_insertions =
            capacity.saturating_sub((DEFAULT_RESERVE_FRACTION * capacity as f64).floor() as usize);

        for key in 0..max_insertions {
            assert_eq!(map.insert(key, key), None);
        }

        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = map.insert(max_insertions, max_insertions);
        }));
        assert!(panic_result.is_err());
    }
}

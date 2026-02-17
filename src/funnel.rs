use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

use crate::common::{
    CTRL_EMPTY, CTRL_TOMBSTONE, DEFAULT_RESERVE_FRACTION, Entry, RawSlots, advance_wrapping_index,
    control_fingerprint, find_first_control, find_first_free_control, find_next_control,
    greatest_common_divisor, is_free_control, is_occupied_control, sanitize_reserve_fraction,
};

const MAX_FUNNEL_RESERVE_FRACTION: f64 = 1.0 / 8.0;

#[derive(Debug)]
struct BucketLevel<K, V> {
    slots: RawSlots<Entry<K, V>>,
    controls: Vec<u8>,
    len: usize,
    bucket_size: usize,
    has_holes: bool,
}

impl<K, V> BucketLevel<K, V> {
    fn with_bucket_count(bucket_count: usize, bucket_size: usize) -> Self {
        let total = bucket_count.saturating_mul(bucket_size);
        Self {
            slots: RawSlots::new(total),
            controls: vec![CTRL_EMPTY; total],
            len: 0,
            bucket_size,
            has_holes: false,
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn bucket_count(&self) -> usize {
        if self.bucket_size == 0 {
            0
        } else {
            self.slots.len() / self.bucket_size
        }
    }
}

impl<K, V> Drop for BucketLevel<K, V> {
    fn drop(&mut self) {
        for (idx, &control) in self.controls.iter().enumerate() {
            if is_occupied_control(control) {
                unsafe { self.slots.drop_in_place(idx) };
            }
        }
    }
}

#[derive(Debug)]
struct SpecialArray<K, V> {
    primary_slots: RawSlots<Entry<K, V>>,
    primary_controls: Vec<u8>,
    primary_len: usize,
    primary_has_holes: bool,
    fallback_slots: RawSlots<Entry<K, V>>,
    fallback_controls: Vec<u8>,
    fallback_len: usize,
    fallback_has_holes: bool,
    fallback_bucket_size: usize,
}

impl<K, V> SpecialArray<K, V> {
    fn with_capacity(capacity: usize, primary_probe_limit: usize) -> Self {
        let fallback_bucket_size = (2usize.saturating_mul(primary_probe_limit)).max(2);
        let desired_fallback_capacity = capacity / 2;
        let fallback_bucket_count = desired_fallback_capacity / fallback_bucket_size;
        let fallback_capacity = fallback_bucket_count.saturating_mul(fallback_bucket_size);
        let primary_capacity = capacity.saturating_sub(fallback_capacity);

        Self {
            primary_slots: RawSlots::new(primary_capacity),
            primary_controls: vec![CTRL_EMPTY; primary_capacity],
            primary_len: 0,
            primary_has_holes: false,
            fallback_slots: RawSlots::new(fallback_capacity),
            fallback_controls: vec![CTRL_EMPTY; fallback_capacity],
            fallback_len: 0,
            fallback_has_holes: false,
            fallback_bucket_size,
        }
    }

    #[inline]
    fn fallback_bucket_count(&self) -> usize {
        if self.fallback_bucket_size == 0 {
            0
        } else {
            self.fallback_slots.len() / self.fallback_bucket_size
        }
    }
}

impl<K, V> Drop for SpecialArray<K, V> {
    fn drop(&mut self) {
        for (idx, &control) in self.primary_controls.iter().enumerate() {
            if is_occupied_control(control) {
                unsafe { self.primary_slots.drop_in_place(idx) };
            }
        }
        for (idx, &control) in self.fallback_controls.iter().enumerate() {
            if is_occupied_control(control) {
                unsafe { self.fallback_slots.drop_in_place(idx) };
            }
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

const INITIAL_CAPACITY: usize = 16;

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
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_params(capacity, DEFAULT_RESERVE_FRACTION)
    }

    fn with_params(capacity: usize, reserve_fraction: f64) -> Self {
        Self::with_params_and_hasher(capacity, reserve_fraction, RandomState::new())
    }

    fn with_params_and_hasher(
        capacity: usize,
        reserve_fraction: f64,
        hash_builder: RandomState,
    ) -> Self {
        let reserve_fraction =
            sanitize_reserve_fraction(reserve_fraction).min(MAX_FUNNEL_RESERVE_FRACTION);
        let level_count = compute_level_count(reserve_fraction);
        let bucket_width = compute_bucket_width(reserve_fraction);

        let mut special_capacity =
            choose_special_capacity(capacity, reserve_fraction, bucket_width);
        let mut main_capacity = capacity.saturating_sub(special_capacity);
        let main_remainder = main_capacity % bucket_width;
        if main_remainder != 0 {
            main_capacity = main_capacity.saturating_sub(main_remainder);
            special_capacity = capacity.saturating_sub(main_capacity);
        }

        let total_main_buckets = main_capacity / bucket_width;
        let level_bucket_counts = partition_funnel_buckets(total_main_buckets, level_count);
        let levels = level_bucket_counts
            .into_iter()
            .map(|bucket_count| BucketLevel::with_bucket_count(bucket_count, bucket_width))
            .collect::<Vec<_>>();

        let primary_probe_limit = log_log_probe_limit(capacity);
        let special = SpecialArray::with_capacity(special_capacity, primary_probe_limit);
        let max_insertions =
            capacity.saturating_sub((reserve_fraction * capacity as f64).floor() as usize);

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
            .unwrap_or_else(|| {
                panic!(
                    "FunnelHashMap: no free slot found; resize not yet implemented (capacity={}, len={})",
                    self.capacity, self.len
                )
            });

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
            } => {
                // SAFETY: find returns only occupied slots.
                Some(unsafe { &self.levels[level_idx].slots.get_ref(slot_idx).value })
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                Some(unsafe { &self.special.primary_slots.get_ref(slot_idx).value })
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                Some(unsafe { &self.special.fallback_slots.get_ref(slot_idx).value })
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
            } => Some(unsafe { &mut self.levels[level_idx].slots.get_mut(slot_idx).value }),
            SlotLocation::SpecialPrimary { slot_idx } => {
                Some(unsafe { &mut self.special.primary_slots.get_mut(slot_idx).value })
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                Some(unsafe { &mut self.special.fallback_slots.get_mut(slot_idx).value })
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
                let level = &mut self.levels[level_idx];
                // SAFETY: find returns only occupied slots.
                let removed = unsafe { level.slots.take(slot_idx) };
                level.controls[slot_idx] = CTRL_TOMBSTONE;
                level.len -= 1;
                level.has_holes = true;
                removed
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                let removed = unsafe { self.special.primary_slots.take(slot_idx) };
                self.special.primary_controls[slot_idx] = CTRL_TOMBSTONE;
                self.special.primary_len -= 1;
                self.special.primary_has_holes = true;
                removed
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                let removed = unsafe { self.special.fallback_slots.take(slot_idx) };
                self.special.fallback_controls[slot_idx] = CTRL_TOMBSTONE;
                self.special.fallback_len -= 1;
                self.special.fallback_has_holes = true;
                removed
            }
        };

        self.len -= 1;
        Some(removed_entry.value)
    }

    pub fn clear(&mut self) {
        for level in &mut self.levels {
            for (idx, control) in level.controls.iter_mut().enumerate() {
                if is_occupied_control(*control) {
                    unsafe { level.slots.drop_in_place(idx) };
                }
                *control = CTRL_EMPTY;
            }
            level.len = 0;
            level.has_holes = false;
        }

        for (idx, control) in self.special.primary_controls.iter_mut().enumerate() {
            if is_occupied_control(*control) {
                unsafe { self.special.primary_slots.drop_in_place(idx) };
            }
            *control = CTRL_EMPTY;
        }
        for (idx, control) in self.special.fallback_controls.iter_mut().enumerate() {
            if is_occupied_control(*control) {
                unsafe { self.special.fallback_slots.drop_in_place(idx) };
            }
            *control = CTRL_EMPTY;
        }
        self.special.primary_len = 0;
        self.special.fallback_len = 0;
        self.special.primary_has_holes = false;
        self.special.fallback_has_holes = false;
        self.len = 0;
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
            level.has_holes = false;
        }

        for (idx, control) in self.special.primary_controls.iter_mut().enumerate() {
            if is_occupied_control(*control) {
                let entry = unsafe { self.special.primary_slots.take(idx) };
                entries.push((entry.key, entry.value));
                *control = CTRL_EMPTY;
            }
        }
        self.special.primary_len = 0;
        self.special.primary_has_holes = false;

        for (idx, control) in self.special.fallback_controls.iter_mut().enumerate() {
            if is_occupied_control(*control) {
                let entry = unsafe { self.special.fallback_slots.take(idx) };
                entries.push((entry.key, entry.value));
                *control = CTRL_EMPTY;
            }
        }
        self.special.fallback_len = 0;
        self.special.fallback_has_holes = false;

        self.len = 0;
        self.max_populated_level = 0;

        let hash_builder = std::mem::replace(&mut self.hash_builder, RandomState::new());
        let mut new_map =
            Self::with_params_and_hasher(new_capacity, self.reserve_fraction, hash_builder);

        for (key, value) in entries {
            new_map.insert(key, value);
        }

        *self = new_map;
    }

    #[inline]
    fn replace_existing_value(&mut self, location: SlotLocation, value: V) -> V {
        match location {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => {
                // SAFETY: caller guarantees slot is occupied.
                let entry = unsafe { self.levels[level_idx].slots.get_mut(slot_idx) };
                std::mem::replace(&mut entry.value, value)
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                let entry = unsafe { self.special.primary_slots.get_mut(slot_idx) };
                std::mem::replace(&mut entry.value, value)
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                let entry = unsafe { self.special.fallback_slots.get_mut(slot_idx) };
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
                self.levels[level_idx]
                    .slots
                    .write(slot_idx, Entry { key, value });
                self.levels[level_idx].controls[slot_idx] = key_fingerprint;
                self.levels[level_idx].len += 1;
                if level_idx > self.max_populated_level {
                    self.max_populated_level = level_idx;
                }
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                self.special
                    .primary_slots
                    .write(slot_idx, Entry { key, value });
                self.special.primary_controls[slot_idx] = key_fingerprint;
                self.special.primary_len += 1;
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                self.special
                    .fallback_slots
                    .write(slot_idx, Entry { key, value });
                self.special.fallback_controls[slot_idx] = key_fingerprint;
                self.special.fallback_len += 1;
            }
        }
        self.len += 1;
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
    fn special_primary_probe_params(&self, key_hash: u64, slot_count: usize) -> (usize, usize) {
        let start = (key_hash.rotate_left(11) as usize) % slot_count;
        let mut step = (key_hash.rotate_left(43) as usize) % slot_count;
        if step == 0 {
            step = 1;
        }
        while greatest_common_divisor(step, slot_count) != 1 {
            step += 1;
            if step == slot_count {
                step = 1;
            }
        }
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
        let bucket_start = bucket_idx * level.bucket_size;
        let bucket_end = bucket_start + level.bucket_size;
        let free_offset = find_first_free_control(&level.controls[bucket_start..bucket_end])?;
        Some(bucket_start + free_offset)
    }

    #[inline]
    fn first_free_in_special_primary(&self, key_hash: u64) -> Option<usize> {
        if self.special.primary_slots.len() == 0
            || self.special.primary_len >= self.special.primary_slots.len()
        {
            return None;
        }

        let slot_count = self.special.primary_slots.len();
        let (start, step) = self.special_primary_probe_params(key_hash, slot_count);
        let mut slot_idx = start;
        for _ in 0..self.primary_probe_limit {
            if is_free_control(self.special.primary_controls[slot_idx]) {
                return Some(slot_idx);
            }
            slot_idx = advance_wrapping_index(slot_idx, step, slot_count);
        }
        None
    }

    #[inline]
    fn first_free_in_special_fallback(&self, key_hash: u64) -> Option<usize> {
        if self.special.fallback_slots.len() == 0
            || self.special.fallback_len >= self.special.fallback_slots.len()
        {
            return None;
        }

        let bucket_count = self.special.fallback_bucket_count();
        if bucket_count == 0 {
            return None;
        }

        let bucket_a = self.special_fallback_bucket_a(key_hash, bucket_count);
        let bucket_b = self.special_fallback_bucket_b(key_hash, bucket_count);
        let bucket_size = self.special.fallback_bucket_size;
        let base_a = bucket_a * bucket_size;
        let base_b = bucket_b * bucket_size;

        for offset in 0..bucket_size {
            let idx_a = base_a + offset;
            if is_free_control(self.special.fallback_controls[idx_a]) {
                return Some(idx_a);
            }

            let idx_b = base_b + offset;
            if is_free_control(self.special.fallback_controls[idx_b]) {
                return Some(idx_b);
            }
        }

        None
    }

    #[inline]
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
        let bucket_start = bucket_idx * level.bucket_size;
        let bucket_end = bucket_start + level.bucket_size;
        let controls = &level.controls[bucket_start..bucket_end];

        let searchable_len = if level.has_holes {
            controls.len()
        } else {
            find_first_control(controls, CTRL_EMPTY).unwrap_or(controls.len())
        };

        let mut match_offset = 0usize;
        while let Some(relative_idx) =
            find_next_control(&controls[..searchable_len], key_fingerprint, match_offset)
        {
            let slot_idx = bucket_start + relative_idx;
            // SAFETY: fingerprint match means slot is occupied.
            let entry = unsafe { level.slots.get_ref(slot_idx) };
            if entry.key.borrow() == key {
                return LookupStep::Found(slot_idx);
            }
            match_offset = relative_idx + 1;
        }

        // If no deletions have occurred in this level and the bucket is not
        // full, the key would have been placed here if it belonged to this
        // bucket — it could not have cascaded further. We can stop early.
        if !level.has_holes && searchable_len < controls.len() {
            LookupStep::StopSearch
        } else {
            LookupStep::Continue
        }
    }

    #[inline]
    fn find_in_special_primary<Q>(&self, key_hash: u64, key_fingerprint: u8, key: &Q) -> LookupStep
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if self.special.primary_slots.len() == 0 {
            return LookupStep::Continue;
        }

        let slot_count = self.special.primary_slots.len();
        let (start, step) = self.special_primary_probe_params(key_hash, slot_count);
        let mut slot_idx = start;
        for _ in 0..self.primary_probe_limit.max(1) {
            let control = self.special.primary_controls[slot_idx];
            if control == key_fingerprint {
                // SAFETY: fingerprint match means slot is occupied.
                let entry = unsafe { self.special.primary_slots.get_ref(slot_idx) };
                if entry.key.borrow() == key {
                    return LookupStep::Found(slot_idx);
                }
            }
            if !self.special.primary_has_holes && control == CTRL_EMPTY {
                return LookupStep::StopSearch;
            }
            slot_idx = advance_wrapping_index(slot_idx, step, slot_count);
        }

        LookupStep::Continue
    }

    #[inline]
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
        if self.special.fallback_slots.len() == 0 {
            return None;
        }

        let bucket_count = self.special.fallback_bucket_count();
        if bucket_count == 0 {
            return None;
        }

        let bucket_a = self.special_fallback_bucket_a(key_hash, bucket_count);
        let bucket_b = self.special_fallback_bucket_b(key_hash, bucket_count);
        let bucket_size = self.special.fallback_bucket_size;
        let base_a = bucket_a * bucket_size;
        let base_b = bucket_b * bucket_size;

        for offset in 0..bucket_size {
            let idx_a = base_a + offset;
            let control_a = self.special.fallback_controls[idx_a];
            if control_a == key_fingerprint {
                // SAFETY: fingerprint match means slot is occupied.
                let entry = unsafe { self.special.fallback_slots.get_ref(idx_a) };
                if entry.key.borrow() == key {
                    return Some(idx_a);
                }
            }
            if !self.special.fallback_has_holes && control_a == CTRL_EMPTY {
                return None;
            }

            let idx_b = base_b + offset;
            let control_b = self.special.fallback_controls[idx_b];
            if control_b == key_fingerprint {
                let entry = unsafe { self.special.fallback_slots.get_ref(idx_b) };
                if entry.key.borrow() == key {
                    return Some(idx_b);
                }
            }
            if !self.special.fallback_has_holes && control_b == CTRL_EMPTY {
                return None;
            }
        }

        None
    }

    #[inline]
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
            LookupStep::Found(slot_idx) => {
                return Some(SlotLocation::SpecialPrimary { slot_idx });
            }
            LookupStep::Continue => {}
            LookupStep::StopSearch => return None,
        }

        self.find_in_special_fallback(key_hash, key_fingerprint, key)
            .map(|slot_idx| SlotLocation::SpecialFallback { slot_idx })
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
            if (total_capacity - special_capacity).is_multiple_of(bucket_size) {
                return special_capacity;
            }
        }
    }

    let target = ((5.0 * reserve_fraction * total_capacity as f64) / 8.0)
        .round()
        .clamp(0.0, total_capacity as f64) as usize;

    let mut best_special_capacity = total_capacity % bucket_size;
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
        let special_capacity = map.special.primary_slots.len() + map.special.fallback_slots.len();

        assert_eq!(level_capacity + special_capacity, map.capacity());
        assert_eq!(map.capacity(), capacity);
    }

    #[test]
    fn funnel_bucket_partition_matches_three_quarters_plus_minus_one() {
        let counts = partition_funnel_buckets(300, 20);
        assert_eq!(counts.iter().sum::<usize>(), 300);

        for pair in counts.windows(2) {
            let expected = 3.0 * pair[0] as f64 / 4.0;
            let observed = pair[1] as f64;
            assert!((observed - expected).abs() <= 1.0);
        }
    }

    #[test]
    fn insert_get_and_update_work() {
        let mut map = FunnelHashMap::with_capacity(512);

        for key in 0..120 {
            assert_eq!(map.insert(key, key * 10), None);
        }
        for key in 0..120 {
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

        assert!(map.contains_key("alpha"));
        assert_eq!(map.remove("alpha"), Some(1));
        assert_eq!(map.remove("alpha"), None);

        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.get("beta"), None);
    }

    #[test]
    fn new_starts_with_zero_capacity() {
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
        let capacity = 256;
        let mut map = FunnelHashMap::with_capacity(capacity);

        for key in 0..capacity {
            let _ = map.insert(key, key * 10);
        }

        for key in 0..capacity {
            assert_eq!(map.get(&key), Some(&(key * 10)));
        }

        assert!(map.capacity() > capacity);
    }
}

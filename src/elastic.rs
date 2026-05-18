use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

use crate::common::DefaultHashBuilder;
use crate::common::TryReserveError;
use crate::common::simd::ProbeOps;

use crate::common::{
    config::{DEFAULT_RESERVE_FRACTION, INITIAL_CAPACITY},
    control::{CTRL_EMPTY, CTRL_TOMBSTONE, ControlByte, ControlOps},
    layout::{Entry, GROUP_SIZE, RawTable},
    math::{
        capacity_for, ceil_three_quarters, floor_half_reserve_slots, level_salt, max_insertions,
        round_up_to_pow2_groups, sanitize_reserve_fraction, usize_to_f64,
    },
};

const DEFAULT_PROBE_SCALE: f64 = 16.0;

/// Construction-time tuning for `ElasticHashMap`.
#[derive(Debug, Clone, Copy)]
pub struct ElasticOptions {
    /// Target initial capacity. The map sizes its level partition so
    /// `capacity * (1 - reserve_fraction)` inserts fit before the next resize.
    capacity: usize,
    /// Fraction of slots kept free as headroom. Lower means higher load
    /// factor but more probing on collisions.
    reserve_fraction: f64,
    /// Multiplier on per-level probe budgets. Higher means more thorough
    /// probing within a level before falling through to the next.
    probe_scale: f64,
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

    #[must_use]
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    #[must_use]
    pub fn reserve_fraction(mut self, reserve_fraction: f64) -> Self {
        self.reserve_fraction = reserve_fraction;
        self
    }

    #[must_use]
    pub fn probe_scale(mut self, probe_scale: f64) -> Self {
        self.probe_scale = probe_scale;
        self
    }
}

/// One level in elastic hashing's geometric partition: an independent
/// open-addressed table roughly half the previous level's capacity.
struct Level<K, V> {
    /// `SoA` control bytes + entries.
    table: RawTable<Entry<K, V>>,
    /// Live entry count.
    len: usize,
    /// Per-level salt mixed into key hashes. Hot — read every lookup.
    salt: u64,
    /// `group_count - 1`. `group_count` is pow2 by construction (see
    /// `partition_levels`), so `(idx + delta) & mask` wraps in one op.
    group_count_mask: usize,
    /// Deleted-slot count.
    tombstones: usize,
    /// Cached `floor(reserve * cap / 2)` for the
    /// `current_free_slots > threshold` branch in slot selection.
    half_reserve_slot_threshold: usize,
    /// Probe budget indexed by `free_slots()`.
    limited_probe_budgets: Box<[usize]>,
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
        debug_assert!(
            group_count == 0 || group_count.is_power_of_two(),
            "partition_levels must produce pow2 group_count",
        );
        let limited_probe_budgets =
            build_probe_budgets(capacity, group_count, reserve_fraction, probe_scale);
        Self {
            table,
            len: 0,
            salt: level_salt(level_idx),
            group_count_mask: group_count.wrapping_sub(1),
            tombstones: 0,
            half_reserve_slot_threshold: floor_half_reserve_slots(reserve_fraction, capacity),
            limited_probe_budgets,
        }
    }

    /// Fallible counterpart to [`Level::with_capacity`].
    fn try_with_capacity(
        capacity: usize,
        reserve_fraction: f64,
        probe_scale: f64,
        level_idx: usize,
    ) -> Result<Self, TryReserveError> {
        let table = RawTable::try_new(capacity).map_err(|()| TryReserveError::AllocError)?;
        let group_count = table.group_count();
        let limited_probe_budgets =
            try_build_probe_budgets(capacity, group_count, reserve_fraction, probe_scale)?;
        Ok(Self {
            table,
            len: 0,
            salt: level_salt(level_idx),
            group_count_mask: group_count.wrapping_sub(1),
            tombstones: 0,
            half_reserve_slot_threshold: floor_half_reserve_slots(reserve_fraction, capacity),
            limited_probe_budgets,
        })
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Slots minus live entries. Includes tombstones (reusable on insert).
    #[inline]
    fn free_slots(&self) -> usize {
        self.capacity().saturating_sub(self.len)
    }

    /// Per-fill-level probe budget (tighter as the level fills).
    #[inline]
    fn limited_group_budget(&self) -> usize {
        self.limited_probe_budgets[self.free_slots()]
    }

    /// Triggers a no-grow rehash on remove when tombstones outnumber half
    /// the slots. Keeps probe sequences from degrading after delete-heavy
    /// workloads.
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

/// Open-addressed hash map using elastic hashing.
///
/// Splits capacity across geometrically shrinking `levels` and routes inserts
/// through a `batch_plan`: early batches concentrate on level 0; later
/// batches push toward deeper levels. Lookups probe every level whose
/// `len > 0`. Unlike standard open addressing, expected probe count stays
/// low even at high load.
pub struct ElasticHashMap<K, V> {
    /// Geometrically shrinking partition of capacity.
    levels: Vec<Level<K, V>>,
    /// Total live entries.
    len: usize,
    /// Total slot count across all levels.
    capacity: usize,
    /// Insert count that triggers `resize(2x)`.
    max_insertions: usize,
    /// Slot reserve fraction per level. See `ElasticOptions`.
    reserve_fraction: f64,
    /// Probe-budget multiplier. See `ElasticOptions`.
    probe_scale: f64,
    /// Per-batch insert quota; drives `current_batch_index` advancement.
    batch_plan: Vec<usize>,
    /// Index into `batch_plan`. Selects which level pair new keys target.
    current_batch_index: usize,
    /// Remaining inserts in the current batch before advancing.
    batch_remaining: usize,
    /// Highest level index ever written; bounds the lookup probe loop.
    max_populated_level: usize,
    /// Hash builder. Cloned across resizes to preserve probe sequences.
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
    pub fn with_hasher(hash_builder: DefaultHashBuilder) -> Self {
        Self::with_options_and_hasher(ElasticOptions::default(), hash_builder)
    }

    #[must_use]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: DefaultHashBuilder) -> Self {
        Self::with_options_and_hasher(ElasticOptions::with_capacity(capacity), hash_builder)
    }

    #[must_use]
    pub fn with_options(options: ElasticOptions) -> Self {
        Self::with_options_and_hasher(options, DefaultHashBuilder::default())
    }

    /// Full constructor. `resize` also calls this with the existing
    /// `hash_builder` so all keys keep the same hash sequence across grows.
    #[must_use]
    pub fn with_options_and_hasher(
        options: ElasticOptions,
        hash_builder: DefaultHashBuilder,
    ) -> Self {
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

    /// Grow capacity so at least `additional` more inserts fit without
    /// triggering an internal resize. No-op if already large enough.
    ///
    /// # Panics
    ///
    /// Panics on capacity overflow. Use [`Self::try_reserve`] for fallible
    /// growth.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len.saturating_add(additional);
        if needed <= self.max_insertions {
            return;
        }
        let new_capacity = self.grow_capacity_for(needed).expect("capacity overflow");
        self.resize(new_capacity);
    }

    /// Fallible counterpart to [`Self::reserve`]. Returns
    /// `Err(TryReserveError::CapacityOverflow)` if `self.len + additional`
    /// overflows `usize`, or `Err(TryReserveError::AllocError)` if the
    /// allocator can't grow the table.
    ///
    /// # Errors
    ///
    /// See above.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let needed = self
            .len
            .checked_add(additional)
            .ok_or(TryReserveError::CapacityOverflow)?;
        if needed <= self.max_insertions {
            return Ok(());
        }
        let new_capacity = self
            .grow_capacity_for(needed)
            .ok_or(TryReserveError::CapacityOverflow)?;
        self.try_resize(new_capacity)
    }

    /// Shrinks the capacity as much as possible while preserving all live
    /// entries. Mirrors [`std::collections::HashMap::shrink_to_fit`].
    pub fn shrink_to_fit(&mut self) {
        self.shrink_to(0);
    }

    /// Shrinks the capacity with a lower bound. The table won't shrink below
    /// the larger of `min_capacity` and `self.len`. Mirrors
    /// [`std::collections::HashMap::shrink_to`].
    ///
    /// # Panics
    ///
    /// Panics if `min_capacity` exceeds any representable capacity (i.e.,
    /// no `usize` capacity satisfies `max_insertions(cap) >= min_capacity`).
    pub fn shrink_to(&mut self, min_capacity: usize) {
        if self.len == 0 && min_capacity == 0 {
            if self.capacity > 0 {
                self.resize(0);
            }
            return;
        }
        let lower = self.len.max(min_capacity).max(INITIAL_CAPACITY);
        let new_capacity = capacity_for(INITIAL_CAPACITY, lower, self.reserve_fraction)
            .expect("capacity overflow");
        if new_capacity >= self.capacity {
            return;
        }
        self.resize(new_capacity);
    }

    /// Round up to the smallest capacity whose `max_insertions` accommodates
    /// `needed` live entries. Returns `None` if no representable capacity
    /// suffices. Used by `reserve` / `try_reserve`.
    fn grow_capacity_for(&self, needed: usize) -> Option<usize> {
        capacity_for(
            self.capacity.max(INITIAL_CAPACITY),
            needed,
            self.reserve_fraction,
        )
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

    /// Like [`Self::get`] but returns the stored key alongside its value.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);
        let (level_idx, slot_idx) =
            self.find_slot_indices_with_hash(key, key_hash, key_fingerprint)?;
        let entry = unsafe { self.levels[level_idx].table.get_ref(slot_idx) };
        Some((&entry.key, &entry.value))
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

    /// Returns `N` disjoint mutable references, mirroring
    /// [`std::collections::HashMap::get_disjoint_mut`]: `None` if any key
    /// misses, panic on aliasing.
    ///
    /// # Panics
    ///
    /// If two input keys resolve to the same `(level, slot)` pair.
    pub fn get_disjoint_mut<Q, const N: usize>(&mut self, keys: [&Q; N]) -> Option<[&mut V; N]>
    where
        K: Borrow<Q> + Eq,
        Q: Hash + Eq + ?Sized,
    {
        let mut locations: [(usize, usize); N] = [(0, 0); N];
        for (i, key) in keys.iter().enumerate() {
            let key_hash = self.hash_key(*key);
            let key_fingerprint = ControlOps::control_fingerprint(key_hash);
            locations[i] = self.find_slot_indices_with_hash(*key, key_hash, key_fingerprint)?;
        }

        // O(N^2) alias check; cheaper than a HashSet for the small N
        // (typically <= 16) std::get_disjoint_mut targets.
        for i in 0..N {
            for j in (i + 1)..N {
                assert!(
                    locations[i] != locations[j],
                    "get_disjoint_mut: duplicate keys resolve to the same entry",
                );
            }
        }

        // SAFETY: locations are unique (checked above) and point to occupied
        // slots. Raw pointer into `levels` lets us hand out disjoint borrows
        // without reborrowing the slice each iteration.
        let levels_ptr: *mut Level<K, V> = self.levels.as_mut_ptr();
        let mut out: core::mem::MaybeUninit<[&mut V; N]> = core::mem::MaybeUninit::uninit();
        let out_ptr = out.as_mut_ptr().cast::<&mut V>();
        for (i, (level_idx, slot_idx)) in locations.into_iter().enumerate() {
            let level = unsafe { &mut *levels_ptr.add(level_idx) };
            let value_ref: &mut V = unsafe { &mut level.table.get_mut(slot_idx).value };
            unsafe { out_ptr.add(i).write(value_ref) };
        }
        Some(unsafe { out.assume_init() })
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
        self.remove_inner(key).map(|(_, v)| v)
    }

    /// Like [`Self::remove`] but returns the stored key alongside its value.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_inner(key)
    }

    fn remove_inner<Q>(&mut self, key: &Q) -> Option<(K, V)>
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
        Some((removed_entry.key, removed_entry.value))
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

    /// Borrowing iterator over `&K`. Order matches [`Self::iter`].
    #[must_use]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Borrowing iterator over `&V`. Order matches [`Self::iter`].
    #[must_use]
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    /// Reference to the map's [`BuildHasher`].
    #[must_use]
    pub fn hasher(&self) -> &DefaultHashBuilder {
        &self.hash_builder
    }
}

/// Borrowing iterator over occupied entries. Walks levels in order, scanning
/// each level's slot array linearly. Skips FREE and TOMBSTONE control bytes.
#[derive(Clone)]
pub struct ElasticIter<'a, K, V> {
    levels: &'a [Level<K, V>],
    level_idx: usize,
    slot_idx: usize,
}

impl<K, V> std::fmt::Debug for ElasticIter<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElasticIter")
            .field("level_idx", &self.level_idx)
            .field("slot_idx", &self.slot_idx)
            .finish_non_exhaustive()
    }
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

/// Borrowing iterator over `&K`, returned by [`ElasticHashMap::keys`].
#[derive(Clone)]
pub struct Keys<'a, K, V> {
    inner: ElasticIter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
}

impl<K, V> std::fmt::Debug for Keys<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keys").finish_non_exhaustive()
    }
}

/// Borrowing iterator over `&V`, returned by [`ElasticHashMap::values`].
#[derive(Clone)]
pub struct Values<'a, K, V> {
    inner: ElasticIter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
}

impl<K, V> std::fmt::Debug for Values<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Values").finish_non_exhaustive()
    }
}

impl<K, V> ElasticHashMap<K, V>
where
    K: Eq + Hash,
{
    /// Drain all live entries into a temp Vec, build a fresh map at
    /// `new_capacity`, reinsert. Passing the current capacity performs a
    /// no-grow rehash that flushes accumulated tombstones.
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

    /// Fallible counterpart to [`Self::resize`] used by `try_reserve`.
    ///
    /// Builds the new (empty) backing storage with fallible allocation
    /// *before* touching `self`, so an `Err` return leaves the map intact.
    fn try_resize(&mut self, new_capacity: usize) -> Result<(), TryReserveError> {
        let hash_builder = self.hash_builder.clone();
        let mut new_map = Self::try_with_options_and_hasher(
            ElasticOptions {
                capacity: new_capacity,
                reserve_fraction: self.reserve_fraction,
                probe_scale: self.probe_scale,
            },
            hash_builder,
        )?;

        for level in &mut self.levels {
            for idx in 0..level.table.capacity() {
                if level.table.control_at(idx).is_occupied() {
                    let entry = unsafe { level.table.take(idx) };
                    new_map.insert(entry.key, entry.value);
                }
            }
            level.table.clear_all_controls();
            level.len = 0;
            level.tombstones = 0;
        }

        self.len = 0;
        self.max_populated_level = 0;
        *self = new_map;
        Ok(())
    }

    /// Fallible counterpart to [`Self::with_options_and_hasher`]. Returns
    /// `Err(TryReserveError::AllocError)` if any backing allocation fails.
    fn try_with_options_and_hasher(
        options: ElasticOptions,
        hash_builder: DefaultHashBuilder,
    ) -> Result<Self, TryReserveError> {
        let reserve_fraction = sanitize_reserve_fraction(options.reserve_fraction);
        let probe_scale = sanitize_probe_scale(options.probe_scale);
        let capacity = options.capacity;
        let max_insertions = max_insertions(capacity, reserve_fraction);

        let level_capacities = partition_levels(capacity);
        let mut levels: Vec<Level<K, V>> = Vec::new();
        levels
            .try_reserve_exact(level_capacities.len())
            .map_err(|_| TryReserveError::AllocError)?;
        for (level_idx, &cap) in level_capacities.iter().enumerate() {
            levels.push(Level::try_with_capacity(
                cap,
                reserve_fraction,
                probe_scale,
                level_idx,
            )?);
        }

        let batch_plan = build_batch_plan(&level_capacities, reserve_fraction, max_insertions);
        let batch_remaining = batch_plan.first().copied().unwrap_or(0);

        Ok(Self {
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
        })
    }

    #[inline]
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        self.hash_builder.hash_one(key)
    }

    /// Advance the batch state machine past any zero-quota batches so the
    /// next insert routes to the correct level pair.
    #[inline]
    fn advance_batch_window(&mut self) {
        while self.batch_remaining == 0 && self.current_batch_index + 1 < self.batch_plan.len() {
            self.current_batch_index += 1;
            self.batch_remaining = self.batch_plan[self.current_batch_index];
        }
    }

    /// Pick the (level, slot) pair to write a new key into. Tries the
    /// batch-targeted level pair first (`choose_slot_targeted`); falls back
    /// to a full sweep across all levels when the targeted slot is full
    /// (e.g. tombstones in earlier levels are the only reusable slots).
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

    /// Batch-driven slot selection. Reads `current_batch_index` to pick the
    /// level pair `(li, li+1)`, then steers between them based on
    /// `current_free_slots > half_reserve_threshold` and `next_free_slots`
    /// thresholds. Per the elastic-hashing schedule, this is what keeps
    /// expected probe count low at high load.
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

    /// Locate `key` across all populated levels. Returns `(level, slot)` on
    /// hit. Bounded by `max_populated_level + 1` so empty trailing levels
    /// don't get probed.
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

    /// Probe one level for `key`. Walks groups via the level's intra-level
    /// probe sequence (triangular for power-of-2 group counts, double-hash
    /// step otherwise), SIMD-matches the fingerprint byte, then key-compares
    /// only the matched slots. Stops on FREE byte (group has space) when no
    /// tombstones exist.
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
        if level.len == 0 {
            return None;
        }

        let group_count = level.table.group_count();
        let mask = level.group_count_mask;
        let capacity = level.capacity();
        let mut group_idx = Self::triangular_group_start(level, key_hash);
        let mut delta: usize = 0;

        for _ in 0..group_count {
            let match_mask = level.table.group_match_mask(group_idx, key_fingerprint);
            for relative_idx in match_mask {
                let slot_idx = group_idx * GROUP_SIZE + relative_idx;
                let entry = unsafe { level.table.get_ref(slot_idx) };
                if entry.key.borrow() == key {
                    return Some(slot_idx);
                }
            }
            let empty_mask = level.table.group_match_mask(group_idx, CTRL_EMPTY);
            if let Some(off) = empty_mask.lowest()
                && group_idx * GROUP_SIZE + off < capacity
            {
                return None;
            }
            delta += 1;
            group_idx = (group_idx + delta) & mask;
        }
        None
    }

    /// Probe-bounded variant of `first_free_uniform`: scans at most
    /// `max_groups` groups. Used by the elastic schedule when
    /// `current_level` still has reserve headroom.
    fn first_free_limited(
        &self,
        key_hash: u64,
        level_idx: usize,
        max_groups: usize,
    ) -> Option<usize> {
        let level = &self.levels[level_idx];
        if level.len >= level.capacity() {
            return None;
        }

        let group_count = level.table.group_count();
        let max_groups = max_groups.min(group_count.max(1));
        let mask = level.group_count_mask;
        let mut group_idx = Self::triangular_group_start(level, key_hash);
        let mut delta: usize = 0;
        for _ in 0..max_groups {
            if let Some(slot_idx) = level.table.first_free_in_group(group_idx) {
                return Some(slot_idx);
            }
            delta += 1;
            group_idx = (group_idx + delta) & mask;
        }
        None
    }

    /// Triangular scan over all groups for the first FREE-or-TOMBSTONE slot.
    /// Returns `None` only if the level is completely OCCUPIED.
    fn first_free_uniform(&self, key_hash: u64, level_idx: usize) -> Option<usize> {
        let level = &self.levels[level_idx];
        if level.len >= level.capacity() {
            return None;
        }

        let group_count = level.table.group_count();
        let mask = level.group_count_mask;
        let mut group_idx = Self::triangular_group_start(level, key_hash);
        let mut delta: usize = 0;
        for _ in 0..group_count {
            if let Some(slot_idx) = level.table.first_free_in_group(group_idx) {
                return Some(slot_idx);
            }
            delta += 1;
            group_idx = (group_idx + delta) & mask;
        }
        None
    }

    /// Triangular-probing starting group: `(key_hash ^ salt) & (group_count - 1)`.
    /// `group_count` is pow2 by `partition_levels` construction.
    #[inline]
    fn triangular_group_start(level: &Level<K, V>, key_hash: u64) -> usize {
        let group_count = level.table.group_count();
        if group_count <= 1 {
            return 0;
        }
        let mixed = key_hash ^ level.salt;
        ProbeOps::hash_to_usize(mixed) & level.group_count_mask
    }

    /// After a remove, walk down `max_populated_level` past any now-empty
    /// trailing levels so subsequent lookups don't probe them.
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

/// Fallible counterpart to [`build_probe_budgets`]. Returns
/// `Err(TryReserveError::AllocError)` on allocation failure.
fn try_build_probe_budgets(
    capacity: usize,
    group_count: usize,
    reserve_fraction: f64,
    probe_scale: f64,
) -> Result<Box<[usize]>, TryReserveError> {
    let mut budgets: Vec<usize> = Vec::new();
    budgets
        .try_reserve_exact(capacity.saturating_add(1))
        .map_err(|_| TryReserveError::AllocError)?;
    budgets.resize(capacity.saturating_add(1), 1);
    if capacity == 0 {
        return Ok(budgets.into_boxed_slice());
    }
    fill_probe_budgets(
        &mut budgets,
        capacity,
        group_count,
        reserve_fraction,
        probe_scale,
    );
    Ok(budgets.into_boxed_slice())
}

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
    fill_probe_budgets(
        &mut budgets,
        capacity,
        group_count,
        reserve_fraction,
        probe_scale,
    );
    budgets.into_boxed_slice()
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn fill_probe_budgets(
    budgets: &mut [usize],
    capacity: usize,
    group_count: usize,
    reserve_fraction: f64,
    probe_scale: f64,
) {
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
}

/// Split `total_capacity` into geometrically halving level sizes, then round
/// each up so `group_count = size / GROUP_SIZE` is pow2 — required for the
/// triangular probe path's `(idx + delta) & mask` wrap. Total slots may
/// exceed `total_capacity` by up to ~2x. Returns `[]` for capacity 0.
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

    sizes.into_iter().map(round_up_to_pow2_groups).collect()
}

/// Build the per-batch insertion quota that drives `current_batch_index`.
/// Batch 0 fills level 0 to ~3/4 occupancy. Each subsequent batch tops up
/// the previous level toward its reserve threshold while priming the next
/// level. Total quota equals `max_insertions`.
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
    fn level_partition_inflates_to_pow2_groups_and_preserves_halving() {
        for &cap in &[127usize, 1_000, 10_000, 100_000] {
            let sizes = partition_levels(cap);
            assert!(!sizes.is_empty());
            // Each level's group_count must be pow2 (triangular precondition).
            for &s in &sizes {
                let g = s / GROUP_SIZE;
                assert!(
                    g.is_power_of_two(),
                    "cap={cap} level slots={s} groups={g} not pow2"
                );
            }
            // Slot total covers the requested capacity, bounded above by 2x.
            let total: usize = sizes.iter().sum();
            assert!(total >= cap, "cap={cap} total={total} below request");
            assert!(total <= cap * 2, "cap={cap} total={total} exceeds 2x");
            // Each next level is at most the previous (non-increasing) and at
            // least half — the geometric halving shape, with pow2 rounding
            // tolerance.
            for w in sizes.windows(2) {
                assert!(w[1] <= w[0], "non-monotonic: {} → {}", w[0], w[1]);
                assert!(w[1] * 2 >= w[0], "shrinks too fast: {} → {}", w[0], w[1]);
            }
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

    #[test]
    fn get_disjoint_mut_returns_all_refs_on_hits() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(64);
        for i in 0..16 {
            map.insert(i, i * 10);
        }

        let got = map.get_disjoint_mut([&1, &3, &7, &15]).expect("all hits");
        assert_eq!(*got[0], 10);
        assert_eq!(*got[1], 30);
        assert_eq!(*got[2], 70);
        assert_eq!(*got[3], 150);
    }

    #[test]
    fn get_disjoint_mut_returns_none_if_any_missing() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(32);
        for i in 0..8 {
            map.insert(i, i);
        }

        assert!(map.get_disjoint_mut([&0, &1, &99]).is_none());
    }

    #[test]
    #[should_panic(expected = "duplicate keys")]
    fn get_disjoint_mut_panics_on_duplicate_keys() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(32);
        map.insert(1, 100);
        map.insert(2, 200);
        let _ = map.get_disjoint_mut([&1, &1]);
    }

    #[test]
    fn get_disjoint_mut_zero_keys_is_some_empty() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(16);
        map.insert(1, 1);
        let got: [&mut i32; 0] = map
            .get_disjoint_mut::<i32, 0>([])
            .expect("zero-key returns Some");
        assert_eq!(got.len(), 0);
    }

    #[test]
    fn get_disjoint_mut_mutation_propagates() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(32);
        for i in 0..8 {
            map.insert(i, i);
        }
        {
            let got = map.get_disjoint_mut([&2, &5]).expect("hit");
            *got[0] = 222;
            *got[1] = 555;
        }
        assert_eq!(map.get(&2), Some(&222));
        assert_eq!(map.get(&5), Some(&555));
    }

    #[test]
    fn keys_yields_inserted_keys_only() {
        use std::collections::HashSet;
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(64);
        for i in 0..30 {
            map.insert(i, i * 10);
        }
        let got: HashSet<i32> = map.keys().copied().collect();
        let expected: HashSet<i32> = (0..30).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn values_yields_inserted_values_only() {
        use std::collections::HashSet;
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(64);
        for i in 0..30 {
            map.insert(i, i * 10);
        }
        let got: HashSet<i32> = map.values().copied().collect();
        let expected: HashSet<i32> = (0..30).map(|i| i * 10).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn hasher_returns_consistent_handle() {
        let map: ElasticHashMap<i32, i32> = ElasticHashMap::new();
        let a: *const _ = map.hasher();
        let b: *const _ = map.hasher();
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn get_key_value_returns_both_on_hit_none_on_miss() {
        let mut map: ElasticHashMap<String, i32> = ElasticHashMap::with_capacity(16);
        map.insert("alpha".to_string(), 1);
        map.insert("beta".to_string(), 2);

        let (k, v) = map.get_key_value("alpha").expect("hit");
        assert_eq!(k, "alpha");
        assert_eq!(*v, 1);

        assert!(map.get_key_value("missing").is_none());
    }

    #[test]
    fn remove_entry_returns_both_and_actually_removes() {
        let mut map: ElasticHashMap<String, i32> = ElasticHashMap::with_capacity(16);
        map.insert("alpha".to_string(), 1);
        map.insert("beta".to_string(), 2);

        let (k, v) = map.remove_entry("alpha").expect("hit");
        assert_eq!(k, "alpha");
        assert_eq!(v, 1);
        assert_eq!(map.len(), 1);
        assert!(map.get("alpha").is_none());

        assert!(map.remove_entry("alpha").is_none());
    }

    #[test]
    fn try_reserve_grows_when_needed() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::new();
        assert_eq!(map.capacity(), 0);
        map.try_reserve(1024).expect("alloc should succeed");
        let cap = map.capacity();
        assert!(cap >= 1024, "reserve under-allocated: cap={cap}");
        for i in 0..1024 {
            map.insert(i, i * 2);
        }
        for i in 0..1024 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
        assert_eq!(map.len(), 1024);
    }

    #[test]
    fn try_reserve_zero_additional_is_noop() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(64);
        let cap_before = map.capacity();
        map.try_reserve(0).expect("noop");
        assert_eq!(map.capacity(), cap_before);
    }

    #[test]
    fn try_reserve_overflow_returns_error() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::new();
        map.insert(1, 1);
        assert_eq!(
            map.try_reserve(usize::MAX),
            Err(TryReserveError::CapacityOverflow)
        );
    }

    #[test]
    fn shrink_to_below_len_clamps_to_len() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(2048);
        for i in 0..100 {
            map.insert(i, i);
        }
        let cap_before = map.capacity();
        map.shrink_to(0);
        let cap_after = map.capacity();
        // Capacity dropped but all entries still queryable.
        assert!(cap_after < cap_before);
        assert!(cap_after >= map.len());
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn shrink_to_above_capacity_is_noop() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(64);
        for i in 0..10 {
            map.insert(i, i);
        }
        let cap = map.capacity();
        map.shrink_to(cap * 4);
        assert_eq!(map.capacity(), cap);
    }

    #[test]
    fn shrink_to_fit_reduces_capacity_when_sparse() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(2048);
        for i in 0..1000 {
            map.insert(i, i);
        }
        for i in 0..900 {
            map.remove(&i);
        }
        let cap_before = map.capacity();
        map.shrink_to_fit();
        assert!(map.capacity() < cap_before);
        for i in 900..1000 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn shrink_then_insert_works() {
        let mut map: ElasticHashMap<i32, i32> = ElasticHashMap::with_capacity(1024);
        for i in 0..200 {
            map.insert(i, i * 3);
        }
        for i in 0..150 {
            map.remove(&i);
        }
        map.shrink_to_fit();
        // Reinserting into a freshly-shrunk map should land cleanly.
        for i in 0..50 {
            assert_eq!(map.insert(i, i * 5), None);
        }
        for i in 0..50 {
            assert_eq!(map.get(&i), Some(&(i * 5)));
        }
        for i in 150..200 {
            assert_eq!(map.get(&i), Some(&(i * 3)));
        }
    }
}

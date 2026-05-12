use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

use crate::common::DefaultHashBuilder;
use crate::common::simd::{ProbeOps, prefetch_read};

use crate::common::{
    config::{DEFAULT_RESERVE_FRACTION, INITIAL_CAPACITY},
    control::{CTRL_EMPTY, ControlByte, ControlOps},
    layout::{Entry, GROUP_SIZE, RawTable},
    math::{
        advance_wrapping_index, ceil_to_usize, fastmod_magic, fastmod_u32, floor_to_usize,
        level_salt, max_insertions, round_to_usize, round_up_to_group, sanitize_reserve_fraction,
        usize_to_f64,
    },
};

pub(crate) const MAX_FUNNEL_RESERVE_FRACTION: f64 = 1.0 / 8.0;

/// Construction-time tuning for `FunnelHashMap`.
#[derive(Debug, Clone, Copy)]
pub struct FunnelOptions {
    /// Target initial capacity. Funnel caps load factor at 1/8 by design;
    /// useful capacity is `capacity * (1 - reserve_fraction)`.
    capacity: usize,
    /// Fraction kept free as headroom. Clamped to
    /// `MAX_FUNNEL_RESERVE_FRACTION` (1/8).
    reserve_fraction: f64,
    /// Max groups probed in the special-array primary before falling back to
    /// the fallback array. `None` derives from `reserve_fraction`.
    primary_probe_limit: Option<usize>,
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
    pub fn primary_probe_limit(mut self, primary_probe_limit: usize) -> Self {
        self.primary_probe_limit = Some(primary_probe_limit);
        self
    }
}

/// One level in the funnel array. Each level is a fixed grid of buckets;
/// inserts hash a key to one bucket and probe within that bucket only.
/// If the bucket is full the insert spills to the next level (or the
/// special array). Bucket-local probing keeps lookup cost bounded.
struct BucketLevel<K, V> {
    /// Structure of Arrays control bytes + entries.
    table: RawTable<Entry<K, V>>,
    /// Live entry count.
    len: usize,
    /// Deleted-slot count.
    tombstones: usize,
    /// Slots per bucket.
    bucket_size: usize,
    /// Number of buckets in this level.
    bucket_count: usize,
    /// Per-level salt mixed into the key hash
    /// so each level distributes differently.
    salt: u64,
    /// Fastmod magic for `bucket_count`.
    bucket_count_magic: u64,
}

impl<K, V> BucketLevel<K, V> {
    fn with_bucket_count(bucket_count: usize, bucket_size: usize, salt: u64) -> Self {
        let total_capacity = bucket_count.saturating_mul(bucket_size);
        let bucket_count_magic = if bucket_count > 1 {
            fastmod_magic(bucket_count)
        } else {
            0
        };
        Self {
            table: RawTable::new(total_capacity),
            len: 0,
            tombstones: 0,
            bucket_size,
            bucket_count,
            salt,
            bucket_count_magic,
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Hash → bucket via fastmod,
    /// salted so each level distributes differently.
    #[inline]
    fn bucket_index(&self, key_hash: u64) -> usize {
        fastmod_u32(
            key_hash ^ self.salt,
            self.bucket_count_magic,
            self.bucket_count,
        )
    }

    /// Slot index range covering all entries in `bucket_idx`.
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

/// Fallback table for keys that didn't fit in any bucket level.
/// Open-addressed with double-hashing across SIMD groups (16 slots each).
struct SpecialPrimary<K, V> {
    /// Structure of Arrays control bytes + entries.
    table: RawTable<Entry<K, V>>,
    /// Live entry count.
    len: usize,
    /// Per-group packed fingerprint metadata for fast scans.
    group_summaries: Box<[u128]>,
    /// Per-group tombstone count, bounds probe length.
    group_tombstones: Box<[usize]>,
    /// Precomputed double-hashing step set.
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
            group_steps: ProbeOps::build_group_steps(group_count),
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

/// Last-resort table for keys that exhaust the special primary's probe
/// budget. Bucketed like `BucketLevel` but with larger buckets (`2 *
/// primary_probe_limit`) so a key that's been pushed this far almost
/// certainly fits.
struct SpecialFallback<K, V> {
    /// Structure of Arrays control bytes + entries.
    table: RawTable<Entry<K, V>>,
    /// Live entry count.
    len: usize,
    /// Deleted-slot count.
    tombstones: usize,
    /// Slots per bucket. Larger than `BucketLevel` (`2 * primary_probe_limit`)
    /// since this is the last-resort table.
    bucket_size: usize,
    /// Number of buckets.
    bucket_count: usize,
}

impl<K, V> SpecialFallback<K, V> {
    fn with_capacity(capacity: usize, bucket_size: usize) -> Self {
        let bucket_count = if bucket_size == 0 {
            0
        } else {
            capacity.div_ceil(bucket_size)
        };
        Self {
            table: RawTable::new(capacity),
            len: 0,
            tombstones: 0,
            bucket_size,
            bucket_count,
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.table.capacity()
    }

    #[inline]
    fn bucket_count(&self) -> usize {
        self.bucket_count
    }

    #[inline]
    fn bucket_range(&self, bucket_idx: usize) -> std::ops::Range<usize> {
        let start = bucket_idx * self.bucket_size;
        let end = (start + self.bucket_size).min(self.table.capacity());
        start..end
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

/// Combines the special primary (probed first) and the special fallback
/// (when primary hits its probe limit). Together they catch keys that
/// overflowed every bucket level.
struct SpecialArray<K, V> {
    /// Probed first; bounded by `primary_probe_limit`.
    primary: SpecialPrimary<K, V>,
    /// Probed after primary hits its limit.
    fallback: SpecialFallback<K, V>,
}

impl<K, V> SpecialArray<K, V> {
    fn with_capacity(capacity: usize, primary_probe_limit: usize) -> Self {
        let fallback_bucket_size = (2usize.saturating_mul(primary_probe_limit)).max(2);
        let primary_capacity = capacity.div_ceil(2);
        let fallback_capacity = capacity.saturating_sub(primary_capacity);
        Self {
            primary: SpecialPrimary::with_capacity(primary_capacity),
            fallback: SpecialFallback::with_capacity(fallback_capacity, fallback_bucket_size),
        }
    }
}

/// Where in the funnel structure a key/slot lives. Returned by lookups,
/// consumed by inserts / removes to avoid recomputing the location.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotLocation {
    Level { level_idx: usize, slot_idx: usize },
    SpecialPrimary { slot_idx: usize },
    SpecialFallback { slot_idx: usize },
}

/// Outcome of probing one bucket / group during lookup.
/// - `Found(slot_idx)`: key matched at slot.
/// - `Continue`: bucket has tombstones; keep probing for the key elsewhere.
/// - `StopSearch`: bucket has free space and no tombstones — key cannot
///   exist further along this hash chain, abort the search.
enum LookupStep {
    Found(usize),
    Continue,
    StopSearch,
}

/// Open-addressed hash map using funnel hashing.
///
/// Capacity is split between a stack of bucket-grouped `levels` (each level
/// half the size of the previous) and a `special` array catching overflow.
/// Inserts try level 0 first, then descend to deeper levels, then to
/// `special.primary`, then `special.fallback`. Lookups follow the same
/// order. The funnel structure trades a small probe budget per level for
/// hard worst-case guarantees on lookup cost.
pub struct FunnelHashMap<K, V> {
    /// Bucket-grouped levels, each half the size of the previous.
    levels: Vec<BucketLevel<K, V>>,
    /// Overflow-catching tables (primary + fallback).
    special: SpecialArray<K, V>,
    /// Total live entries across levels + special.
    len: usize,
    /// Total slot count.
    capacity: usize,
    /// Insert count that triggers `resize(2x)`.
    max_insertions: usize,
    /// Slot reserve fraction. See `FunnelOptions`.
    reserve_fraction: f64,
    /// Cap on groups probed in the special primary before fallback.
    primary_probe_limit: usize,
    /// Highest level index ever written; bounds the lookup probe loop.
    max_populated_level: usize,
    /// Hash builder. Cloned across resizes to preserve probe sequences.
    hash_builder: DefaultHashBuilder,
}

impl<K: std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug for FunnelHashMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunnelHashMap")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("max_populated_level", &self.max_populated_level)
            .finish_non_exhaustive()
    }
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
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(FunnelOptions::default())
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_options(FunnelOptions::with_capacity(capacity))
    }

    #[must_use]
    pub fn with_hasher(hash_builder: DefaultHashBuilder) -> Self {
        Self::with_options_and_hasher(FunnelOptions::default(), hash_builder)
    }

    #[must_use]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: DefaultHashBuilder) -> Self {
        Self::with_options_and_hasher(FunnelOptions::with_capacity(capacity), hash_builder)
    }

    #[must_use]
    pub fn with_options(options: FunnelOptions) -> Self {
        Self::with_options_and_hasher(options, DefaultHashBuilder::default())
    }

    /// Full constructor. `resize` also calls this with the existing
    /// `hash_builder` so all keys keep the same hash sequence across grows.
    #[must_use]
    pub fn with_options_and_hasher(
        options: FunnelOptions,
        hash_builder: DefaultHashBuilder,
    ) -> Self {
        let reserve_fraction =
            sanitize_reserve_fraction(options.reserve_fraction).min(MAX_FUNNEL_RESERVE_FRACTION);
        let capacity = options.capacity;
        let max_insertions = max_insertions(capacity, reserve_fraction);

        let level_count = compute_level_count(reserve_fraction);
        let bucket_width = round_up_to_group(compute_bucket_width(reserve_fraction));
        let primary_probe_limit = options
            .primary_probe_limit
            .unwrap_or_else(|| ProbeOps::log_log_probe_limit(capacity))
            .max(1);

        let mut special_capacity =
            choose_special_capacity(capacity, reserve_fraction, bucket_width);
        let mut main_capacity = capacity.saturating_sub(special_capacity);
        let main_remainder = main_capacity % bucket_width.max(1);
        if main_remainder != 0 {
            main_capacity = main_capacity.saturating_sub(main_remainder);
            special_capacity = capacity.saturating_sub(main_capacity);
        }

        let total_main_buckets = main_capacity.checked_div(bucket_width).unwrap_or(0);
        let level_bucket_counts = partition_funnel_buckets(total_main_buckets, level_count);
        let levels = level_bucket_counts
            .into_iter()
            .enumerate()
            .map(|(level_idx, bucket_count)| {
                BucketLevel::with_bucket_count(bucket_count, bucket_width, level_salt(level_idx))
            })
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
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len.saturating_add(additional);
        if needed <= self.max_insertions {
            return;
        }
        let mut new_capacity = self.capacity.max(INITIAL_CAPACITY);
        while max_insertions(new_capacity, self.reserve_fraction) < needed {
            new_capacity = new_capacity.saturating_mul(2);
        }
        self.resize(new_capacity);
    }

    /// # Panics
    ///
    /// Panics if a resize succeeds but no free slot can be found for the new key.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let key_hash = self.hash_key(&key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);

        let level_candidate =
            match self.find_in_levels_with_candidate(&key, key_hash, key_fingerprint) {
                Ok(location) => return Some(self.replace_existing_value(location, value)),
                Err(level_candidate) => level_candidate,
            };

        if let Some(location) = level_candidate
            && self.special.primary.len == 0
            && self.special.fallback.len == 0
        {
            return self.insert_at_location_after_resize_check(
                Some(location),
                key_hash,
                key,
                value,
                key_fingerprint,
            );
        }

        let (primary_step, primary_candidate) =
            self.find_in_special_primary_with_candidate(key_hash, key_fingerprint, &key);
        match primary_step {
            LookupStep::Found(slot_idx) => {
                return Some(
                    self.replace_existing_value(SlotLocation::SpecialPrimary { slot_idx }, value),
                );
            }
            LookupStep::StopSearch => {
                if let Some(location) = level_candidate {
                    return self.insert_at_location_after_resize_check(
                        Some(location),
                        key_hash,
                        key,
                        value,
                        key_fingerprint,
                    );
                }
                return self.insert_at_location_after_resize_check(
                    primary_candidate.map(|slot_idx| SlotLocation::SpecialPrimary { slot_idx }),
                    key_hash,
                    key,
                    value,
                    key_fingerprint,
                );
            }
            LookupStep::Continue => {}
        }

        let (fallback_match, fallback_candidate) =
            self.find_in_special_fallback_with_candidate(key_hash, key_fingerprint, &key);
        if let Some(slot_idx) = fallback_match {
            return Some(
                self.replace_existing_value(SlotLocation::SpecialFallback { slot_idx }, value),
            );
        }

        let insertion_slot = level_candidate
            .or_else(|| primary_candidate.map(|slot_idx| SlotLocation::SpecialPrimary { slot_idx }))
            .or_else(|| {
                fallback_candidate.map(|slot_idx| SlotLocation::SpecialFallback { slot_idx })
            });
        self.insert_at_location_after_resize_check(
            insertion_slot,
            key_hash,
            key,
            value,
            key_fingerprint,
        )
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);

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

    /// Batched lookup: pipelines N keys by issuing prefetches for the
    /// level-0 bucket `PIPELINE_DEPTH` iterations ahead of the resolution
    /// loop. Overlaps independent DRAM/L3 misses.
    ///
    /// Allocates a fresh `Vec<Option<&V>>` on every call. Callers that
    /// re-issue batches in a hot loop should prefer
    /// [`Self::multi_get_into`], which writes into a caller-owned buffer.
    ///
    /// # Prefetch scope
    ///
    /// Only the level-0 bucket group is prefetched. Miss-heavy batches that
    /// probe into level >= 1 or fall through to the special arrays see no
    /// prefetch benefit and may pay one speculative L1 fetch per key.
    pub fn multi_get<'a, Q>(&self, keys: &[&'a Q]) -> Vec<Option<&V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'a,
    {
        let mut out = Vec::with_capacity(keys.len());
        self.multi_get_into(keys, &mut out);
        out
    }

    /// Sibling of [`Self::multi_get`] that writes results into `out`,
    /// reusing its allocation across calls. `out` is `clear`ed first and
    /// reserved to hold exactly `keys.len()` entries. Same pipelined
    /// prefetch and same prefetch-scope caveats as `multi_get`.
    pub fn multi_get_into<'a, 'b, Q>(&'a self, keys: &[&'b Q], out: &mut Vec<Option<&'a V>>)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'b,
    {
        // See ElasticHashMap::multi_get_into for depth tuning rationale.
        const PIPELINE_DEPTH: usize = 8;

        let n = keys.len();
        out.clear();
        out.reserve(n);
        if self.len == 0 {
            out.extend(std::iter::repeat_n(None, n));
            return;
        }

        let hashes: Vec<u64> = keys.iter().map(|k| self.hash_key(*k)).collect();

        let level0_opt = self.levels.first().filter(|l| l.len > 0);

        if let Some(level0) = level0_opt {
            for &h in hashes.iter().take(PIPELINE_DEPTH.min(n)) {
                let bucket_idx = level0.bucket_index(h);
                let group_idx = (bucket_idx * level0.bucket_size) / GROUP_SIZE;
                unsafe { prefetch_read(level0.table.group_data_ptr(group_idx)) };
            }
        }

        for i in 0..n {
            if let Some(level0) = level0_opt
                && let Some(&h_ahead) = hashes.get(i + PIPELINE_DEPTH)
            {
                let bucket_idx = level0.bucket_index(h_ahead);
                let group_idx = (bucket_idx * level0.bucket_size) / GROUP_SIZE;
                unsafe { prefetch_read(level0.table.group_data_ptr(group_idx)) };
            }

            let h = hashes[i];
            let fp = ControlOps::control_fingerprint(h);
            let result = self
                .find_slot_location_with_hash(keys[i], h, fp)
                .map(|loc| match loc {
                    SlotLocation::Level {
                        level_idx,
                        slot_idx,
                    } => unsafe { &self.levels[level_idx].table.get_ref(slot_idx).value },
                    SlotLocation::SpecialPrimary { slot_idx } => unsafe {
                        &self.special.primary.table.get_ref(slot_idx).value
                    },
                    SlotLocation::SpecialFallback { slot_idx } => unsafe {
                        &self.special.fallback.table.get_ref(slot_idx).value
                    },
                });
            out.push(result);
        }
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);

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

    /// Returns `N` disjoint mutable references to the values for `keys`.
    ///
    /// Matches the semantics of `std::collections::HashMap::get_disjoint_mut`:
    /// returns `None` if any key is missing, and panics if any two keys
    /// resolve to the same slot. Probes run sequentially because mutable
    /// refs can only be materialized after the alias check completes, so
    /// the pipelined `multi_get` path does not apply here.
    ///
    /// # Panics
    ///
    /// Panics if two input keys resolve to the same physical slot
    /// (i.e. they refer to the same entry).
    pub fn get_disjoint_mut<Q, const N: usize>(&mut self, keys: [&Q; N]) -> Option<[&mut V; N]>
    where
        K: Borrow<Q> + Eq,
        Q: Hash + Eq + ?Sized,
    {
        // Resolve each key. Bail on first miss.
        let mut locations: [SlotLocation; N] = [SlotLocation::SpecialPrimary { slot_idx: 0 }; N];
        for (i, key) in keys.iter().enumerate() {
            let key_hash = self.hash_key(*key);
            let key_fingerprint = ControlOps::control_fingerprint(key_hash);
            locations[i] = self.find_slot_location_with_hash(*key, key_hash, key_fingerprint)?;
        }

        // O(N^2) alias check.
        for i in 0..N {
            for j in (i + 1)..N {
                assert!(
                    locations[i] != locations[j],
                    "get_disjoint_mut: duplicate keys resolve to the same entry",
                );
            }
        }

        // Materialize mutable refs into the result array. SAFETY: every
        // resolved location is unique and points to an occupied slot. Raw
        // pointers let us hand out disjoint borrows into `levels` /
        // `special` without re-borrowing the slices each iteration.
        let levels_ptr: *mut BucketLevel<K, V> = self.levels.as_mut_ptr();
        let primary_ptr: *mut SpecialPrimary<K, V> = &raw mut self.special.primary;
        let fallback_ptr: *mut SpecialFallback<K, V> = &raw mut self.special.fallback;
        let mut out: core::mem::MaybeUninit<[&mut V; N]> = core::mem::MaybeUninit::uninit();
        let out_ptr = out.as_mut_ptr().cast::<&mut V>();
        for (i, loc) in locations.into_iter().enumerate() {
            let value_ref: &mut V = match loc {
                SlotLocation::Level {
                    level_idx,
                    slot_idx,
                } => unsafe {
                    let level = &mut *levels_ptr.add(level_idx);
                    &mut level.table.get_mut(slot_idx).value
                },
                SlotLocation::SpecialPrimary { slot_idx } => unsafe {
                    &mut (*primary_ptr).table.get_mut(slot_idx).value
                },
                SlotLocation::SpecialFallback { slot_idx } => unsafe {
                    &mut (*fallback_ptr).table.get_mut(slot_idx).value
                },
            };
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
        self.find_slot_location_with_hash(key, key_hash, key_fingerprint)
            .is_some()
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let key_hash = self.hash_key(key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);
        let location = self.find_slot_location_with_hash(key, key_hash, key_fingerprint)?;

        let (removed_entry, needs_resize) = match location {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => {
                let level = &mut self.levels[level_idx];
                let removed = unsafe { level.table.take(slot_idx) };
                level.table.mark_tombstone(slot_idx);
                level.len -= 1;
                level.tombstones += 1;
                let needs_resize = level.tombstones > level.capacity() / 2;
                (removed, needs_resize)
            }
            SlotLocation::SpecialPrimary { slot_idx } => {
                let group_idx = slot_idx / GROUP_SIZE;
                let primary = &mut self.special.primary;
                let removed = unsafe { primary.table.take(slot_idx) };
                primary.table.mark_tombstone(slot_idx);
                primary.len -= 1;
                primary.group_tombstones[group_idx] += 1;
                let needs_resize = primary.group_tombstones[group_idx] > GROUP_SIZE / 4;
                (removed, needs_resize)
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                let fallback = &mut self.special.fallback;
                let removed = unsafe { fallback.table.take(slot_idx) };
                fallback.table.mark_tombstone(slot_idx);
                fallback.len -= 1;
                fallback.tombstones += 1;
                let needs_resize = fallback.tombstones > fallback.capacity() / 2;
                (removed, needs_resize)
            }
        };

        self.len -= 1;
        self.shrink_max_populated_level();
        if needs_resize {
            self.resize(self.capacity);
        }
        Some(removed_entry.value)
    }

    #[must_use]
    pub fn iter(&self) -> FunnelIter<'_, K, V> {
        FunnelIter {
            levels: &self.levels,
            primary: &self.special.primary,
            fallback: &self.special.fallback,
            phase: FunnelIterPhase::Levels,
            level_idx: 0,
            slot_idx: 0,
        }
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
        self.special.fallback.tombstones = 0;

        self.len = 0;
        self.max_populated_level = 0;
    }

    /// Drain all live entries (across levels + special), build a fresh map
    /// at `new_capacity`, reinsert. Also serves as a no-grow rehash when
    /// called with the current capacity.
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
        self.special.fallback.tombstones = 0;

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
            new_map.insert_new_entry_unchecked(key, value);
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
    fn special_primary_probe_params(&self, key_hash: u64, group_count: usize) -> (usize, usize) {
        if group_count <= 1 {
            return (0, 1);
        }
        let start = ProbeOps::hash_to_usize(key_hash.rotate_left(11)) % group_count;
        let steps = &self.special.primary.group_steps;
        let step = steps[ProbeOps::hash_to_usize(key_hash.rotate_left(43)) % steps.len()];
        (start, step)
    }

    #[inline]
    fn special_fallback_bucket_a(key_hash: u64, bucket_count: usize) -> usize {
        ProbeOps::hash_to_usize(key_hash.rotate_left(19)) % bucket_count
    }

    #[inline]
    fn special_fallback_bucket_b(key_hash: u64, bucket_count: usize) -> usize {
        ProbeOps::hash_to_usize(key_hash.rotate_left(37)) % bucket_count
    }

    #[inline]
    fn choose_slot_for_new_key(&self, key_hash: u64) -> Option<SlotLocation> {
        for (level_idx, level) in self.levels.iter().enumerate() {
            if let Some(slot_idx) = Self::first_free_in_level_bucket(key_hash, level) {
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

    /// Search bucket levels for `key`. Returns `Ok(SlotLocation)` on hit, or
    /// `Err(Some(insert_location))` with the first non-full bucket's
    /// candidate slot if known (used by insert to skip a re-search).
    /// `Err(None)` when no insert candidate was seen and the search exhausted.
    fn find_in_levels_with_candidate<Q>(
        &self,
        key: &Q,
        key_hash: u64,
        key_fingerprint: u8,
    ) -> Result<SlotLocation, Option<SlotLocation>>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let search_limit = (self.max_populated_level + 1).min(self.levels.len());
        let mut candidate = None;

        for (level_idx, level) in self.levels[..search_limit].iter().enumerate() {
            let (lookup_step, slot_candidate) =
                Self::find_in_level_bucket_with_candidate(key_hash, key_fingerprint, key, level);
            if candidate.is_none() {
                candidate = slot_candidate.map(|slot_idx| SlotLocation::Level {
                    level_idx,
                    slot_idx,
                });
            }
            match lookup_step {
                LookupStep::Found(slot_idx) => {
                    return Ok(SlotLocation::Level {
                        level_idx,
                        slot_idx,
                    });
                }
                LookupStep::Continue => {}
                LookupStep::StopSearch => return Err(candidate),
            }
        }

        Err(candidate)
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
    fn insert_new_entry_unchecked(&mut self, key: K, value: V) {
        let key_hash = self.hash_key(&key);
        let key_fingerprint = ControlOps::control_fingerprint(key_hash);
        let location = self
            .choose_slot_for_new_key(key_hash)
            .expect("resized funnel map should have free slot");
        self.place_new_entry(location, key, value, key_fingerprint);
    }

    #[inline]
    /// Insert `key`/`value` into a candidate slot, growing first via
    /// `resize` if `len >= max_insertions`. After resize, the candidate
    /// becomes stale, so this re-locates the slot from scratch.
    fn insert_at_location_after_resize_check(
        &mut self,
        location: Option<SlotLocation>,
        key_hash: u64,
        key: K,
        value: V,
        key_fingerprint: u8,
    ) -> Option<V> {
        let final_location = if self.len >= self.max_insertions || location.is_none() {
            let new_capacity = if self.capacity == 0 {
                INITIAL_CAPACITY
            } else {
                self.capacity.saturating_mul(2)
            };
            self.resize(new_capacity);
            self.choose_slot_for_new_key(key_hash)
                .expect("no free slot found after resize")
        } else {
            match location {
                Some(location) => location,
                None => unreachable!("checked for resize above"),
            }
        };

        self.place_new_entry(final_location, key, value, key_fingerprint);
        None
    }

    #[inline]
    fn place_new_entry(&mut self, location: SlotLocation, key: K, value: V, key_fingerprint: u8) {
        match location {
            SlotLocation::Level {
                level_idx,
                slot_idx,
            } => {
                let level = &mut self.levels[level_idx];
                level
                    .table
                    .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
                level.len += 1;
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
                primary.group_summaries[group_idx] |= ControlOps::fingerprint_bit(key_fingerprint);
            }
            SlotLocation::SpecialFallback { slot_idx } => {
                let fallback = &mut self.special.fallback;
                fallback
                    .table
                    .write_with_control(slot_idx, Entry { key, value }, key_fingerprint);
                fallback.len += 1;
            }
        }
        self.len += 1;
    }

    fn first_free_in_level_bucket(key_hash: u64, level: &BucketLevel<K, V>) -> Option<usize> {
        if level.len >= level.capacity() || level.bucket_count == 0 {
            return None;
        }

        let bucket_idx = level.bucket_index(key_hash);
        let bucket_range = level.bucket_range(bucket_idx);
        debug_assert_eq!(bucket_range.start % GROUP_SIZE, 0);
        let group_idx = bucket_range.start / GROUP_SIZE;
        level
            .table
            .group_free_mask(group_idx)
            .truncate_to(level.bucket_size)
            .lowest()
            .map(|offset| bucket_range.start + offset)
    }

    fn first_free_in_special_primary(&self, key_hash: u64) -> Option<usize> {
        let primary = &self.special.primary;
        if primary.len >= primary.table.capacity() {
            return None;
        }

        let group_count = primary.table.group_count();
        let (group_start, group_step) = self.special_primary_probe_params(key_hash, group_count);
        let mut group_idx = group_start;
        let group_limit = self.primary_probe_limit.min(group_count.max(1));

        for _ in 0..group_limit {
            if let Some(slot_idx) = Self::first_free_in_special_primary_group(primary, group_idx) {
                return Some(slot_idx);
            }
            group_idx = advance_wrapping_index(group_idx, group_step, group_count);
        }
        None
    }

    fn first_free_in_special_fallback(&self, key_hash: u64) -> Option<usize> {
        let fallback = &self.special.fallback;
        if fallback.len >= fallback.capacity() {
            return None;
        }

        let bucket_count = fallback.bucket_count();
        if bucket_count == 0 {
            return None;
        }

        let bucket_a = Self::special_fallback_bucket_a(key_hash, bucket_count);
        let bucket_b = Self::special_fallback_bucket_b(key_hash, bucket_count);

        for &bucket_idx in &[bucket_a, bucket_b] {
            let range = fallback.bucket_range(bucket_idx);
            for slot_idx in range {
                if fallback.table.control_at(slot_idx).is_free() {
                    return Some(slot_idx);
                }
            }
        }

        None
    }

    #[inline]
    fn first_free_in_special_primary_group(
        primary: &SpecialPrimary<K, V>,
        group_idx: usize,
    ) -> Option<usize> {
        primary.table.first_free_in_group(group_idx)
    }

    fn find_in_level_bucket<Q>(
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
        level: &BucketLevel<K, V>,
    ) -> LookupStep
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if level.len == 0 || level.bucket_count == 0 {
            return LookupStep::Continue;
        }

        let bucket_idx = level.bucket_index(key_hash);
        let bucket_range = level.bucket_range(bucket_idx);
        debug_assert_eq!(bucket_range.start % GROUP_SIZE, 0);
        let group_idx = bucket_range.start / GROUP_SIZE;

        // SIMD fingerprint scan — same as _with_candidate.
        for relative_idx in level
            .table
            .group_match_mask(group_idx, key_fingerprint)
            .truncate_to(level.bucket_size)
        {
            let slot_idx = bucket_range.start + relative_idx;
            let entry = unsafe { level.table.get_ref(slot_idx) };
            if entry.key.borrow() == key {
                return LookupStep::Found(slot_idx);
            }
        }

        // StopSearch: bucket has an EMPTY byte → no key ever overflowed past
        // here. Tombstones in the bucket don't disable termination since the
        // empty byte still proves the probe chain terminated naturally.
        if level
            .table
            .group_match_mask(group_idx, CTRL_EMPTY)
            .truncate_to(level.bucket_size)
            .any()
        {
            return LookupStep::StopSearch;
        }

        LookupStep::Continue
    }

    fn find_in_level_bucket_with_candidate<Q>(
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
        level: &BucketLevel<K, V>,
    ) -> (LookupStep, Option<usize>)
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if level.len == 0 {
            return (LookupStep::Continue, None);
        }

        if level.bucket_count == 0 {
            return (LookupStep::Continue, None);
        }

        let bucket_idx = level.bucket_index(key_hash);
        let bucket_range = level.bucket_range(bucket_idx);
        debug_assert_eq!(bucket_range.start % GROUP_SIZE, 0);
        let group_idx = bucket_range.start / GROUP_SIZE;

        // SIMD fingerprint scan over the bucket's control bytes.
        for relative_idx in level
            .table
            .group_match_mask(group_idx, key_fingerprint)
            .truncate_to(level.bucket_size)
        {
            let slot_idx = bucket_range.start + relative_idx;
            let entry = unsafe { level.table.get_ref(slot_idx) };
            if entry.key.borrow() == key {
                let free_candidate = level
                    .table
                    .group_free_mask(group_idx)
                    .truncate_to(level.bucket_size)
                    .lowest()
                    .map(|o| bucket_range.start + o);
                return (LookupStep::Found(slot_idx), free_candidate);
            }
        }

        // No match — compute free candidate and early-exit status.
        let free_candidate = level
            .table
            .group_free_mask(group_idx)
            .truncate_to(level.bucket_size)
            .lowest()
            .map(|o| bucket_range.start + o);

        let step = if level
            .table
            .group_match_mask(group_idx, CTRL_EMPTY)
            .truncate_to(level.bucket_size)
            .any()
        {
            LookupStep::StopSearch
        } else {
            LookupStep::Continue
        };

        (step, free_candidate)
    }

    /// Probe the special primary for `key` (lookup-only — no insert
    /// candidate tracking). Bounded by `primary_probe_limit` groups; if
    /// reached without a match and no tombstones seen, returns `StopSearch`
    /// so the caller skips fallback.
    fn find_in_special_primary<Q>(&self, key_hash: u64, key_fingerprint: u8, key: &Q) -> LookupStep
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let primary = &self.special.primary;
        if primary.len == 0 {
            return LookupStep::Continue;
        }

        let fingerprint_mask = ControlOps::fingerprint_bit(key_fingerprint);
        let group_count = primary.table.group_count();
        let (group_start, group_step) = self.special_primary_probe_params(key_hash, group_count);
        let mut group_idx = group_start;
        let group_limit = self.primary_probe_limit.min(group_count.max(1));

        for _ in 0..group_limit {
            if primary.group_summaries[group_idx] & fingerprint_mask != 0 {
                for relative_idx in primary.table.group_match_mask(group_idx, key_fingerprint) {
                    let slot_idx = group_idx * GROUP_SIZE + relative_idx;
                    let entry = unsafe { primary.table.get_ref(slot_idx) };
                    if entry.key.borrow() == key {
                        return LookupStep::Found(slot_idx);
                    }
                }
            }

            if primary.group_tombstones[group_idx] == 0
                && primary.table.first_free_in_group(group_idx).is_some()
            {
                return LookupStep::StopSearch;
            }

            let next = advance_wrapping_index(group_idx, group_step, group_count);
            unsafe { prefetch_read(primary.table.group_data_ptr(next)) };
            group_idx = next;
        }

        LookupStep::Continue
    }

    /// Like `find_in_special_primary`, but also remembers the first
    /// FREE-or-TOMBSTONE slot seen so insert can land there without a re-scan.
    fn find_in_special_primary_with_candidate<Q>(
        &self,
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
    ) -> (LookupStep, Option<usize>)
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let primary = &self.special.primary;
        if primary.table.capacity() == 0 {
            return (LookupStep::Continue, None);
        }
        if primary.len == 0 {
            return (
                LookupStep::Continue,
                self.first_free_in_special_primary(key_hash),
            );
        }

        let fingerprint_mask = ControlOps::fingerprint_bit(key_fingerprint);
        let group_count = primary.table.group_count();
        let (group_start, group_step) = self.special_primary_probe_params(key_hash, group_count);
        let mut group_idx = group_start;
        let group_limit = self.primary_probe_limit.min(group_count.max(1));
        let mut candidate = None;

        for _ in 0..group_limit {
            if candidate.is_none() && primary.table.first_free_in_group(group_idx).is_some() {
                candidate = primary.table.first_free_in_group(group_idx);
            }

            if primary.group_summaries[group_idx] & fingerprint_mask != 0 {
                for relative_idx in primary.table.group_match_mask(group_idx, key_fingerprint) {
                    let slot_idx = group_idx * GROUP_SIZE + relative_idx;
                    let entry = unsafe { primary.table.get_ref(slot_idx) };
                    if entry.key.borrow() == key {
                        return (LookupStep::Found(slot_idx), candidate);
                    }
                }
            }

            if primary.group_tombstones[group_idx] == 0
                && primary.table.first_free_in_group(group_idx).is_some()
            {
                return (LookupStep::StopSearch, candidate);
            }

            let next = advance_wrapping_index(group_idx, group_step, group_count);
            unsafe { prefetch_read(primary.table.group_data_ptr(next)) };
            group_idx = next;
        }

        (LookupStep::Continue, candidate)
    }

    /// Probe the special fallback for `key`. Bucket-local search like
    /// `BucketLevel`, but with larger buckets sized for primary spillover.
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
        if fallback.len == 0 {
            return None;
        }

        let bucket_count = fallback.bucket_count();
        if bucket_count == 0 {
            return None;
        }

        let bucket_a = Self::special_fallback_bucket_a(key_hash, bucket_count);
        let bucket_b = Self::special_fallback_bucket_b(key_hash, bucket_count);

        for bucket_idx in [bucket_a, bucket_b] {
            let range = fallback.bucket_range(bucket_idx);
            let controls = unsafe {
                std::slice::from_raw_parts(
                    fallback.table.group_data_ptr(0).add(range.start),
                    range.len(),
                )
            };

            let mut match_offset = 0;
            while let Some(relative_idx) = ControlOps::find_next_fingerprint_in_controls(
                controls,
                key_fingerprint,
                match_offset,
            ) {
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

    /// Like `find_in_special_fallback`, but also tracks the first
    /// FREE-or-TOMBSTONE slot for insert.
    fn find_in_special_fallback_with_candidate<Q>(
        &self,
        key_hash: u64,
        key_fingerprint: u8,
        key: &Q,
    ) -> (Option<usize>, Option<usize>)
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let fallback = &self.special.fallback;
        if fallback.capacity() == 0 {
            return (None, None);
        }
        if fallback.len == 0 {
            return (None, self.first_free_in_special_fallback(key_hash));
        }

        let bucket_count = fallback.bucket_count();
        if bucket_count == 0 {
            return (None, None);
        }

        let bucket_a = Self::special_fallback_bucket_a(key_hash, bucket_count);
        let bucket_b = Self::special_fallback_bucket_b(key_hash, bucket_count);
        let mut candidate = None;

        // Find a free slot in either bucket.
        for &bucket_idx in &[bucket_a, bucket_b] {
            if candidate.is_some() {
                break;
            }
            let range = fallback.bucket_range(bucket_idx);
            for slot_idx in range {
                if fallback.table.control_at(slot_idx).is_free() {
                    candidate = Some(slot_idx);
                    break;
                }
            }
        }

        (
            self.find_in_special_fallback(key_hash, key_fingerprint, key),
            candidate,
        )
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
            match Self::find_in_level_bucket(key_hash, key_fingerprint, key, level) {
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

        if self.special.primary.len == 0 && self.special.fallback.len == 0 {
            return None;
        }

        match self.find_in_special_primary(key_hash, key_fingerprint, key) {
            LookupStep::Found(slot_idx) => return Some(SlotLocation::SpecialPrimary { slot_idx }),
            LookupStep::Continue => {}
            LookupStep::StopSearch => return None,
        }

        self.find_in_special_fallback(key_hash, key_fingerprint, key)
            .map(|slot_idx| SlotLocation::SpecialFallback { slot_idx })
    }

    fn shrink_max_populated_level(&mut self) {
        while self.max_populated_level > 0 && self.levels[self.max_populated_level].len == 0 {
            self.max_populated_level -= 1;
        }
        if self.levels.is_empty() || self.levels[0].len == 0 {
            self.max_populated_level = 0;
        }
    }
}

/// Three-phase iterator state: walk all bucket levels, then the special
/// primary, then the special fallback.
enum FunnelIterPhase {
    Levels,
    Primary,
    Fallback,
    Done,
}

/// Borrowing iterator over occupied entries.
/// Visits each region in funnel order: bucket levels → special primary → special fallback.
/// Skips FREE and TOMBSTONE control bytes.
pub struct FunnelIter<'a, K, V> {
    levels: &'a [BucketLevel<K, V>],
    primary: &'a SpecialPrimary<K, V>,
    fallback: &'a SpecialFallback<K, V>,
    phase: FunnelIterPhase,
    level_idx: usize,
    slot_idx: usize,
}

impl<'a, K, V> Iterator for FunnelIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.phase {
                FunnelIterPhase::Levels => {
                    while self.level_idx < self.levels.len() {
                        let table = &self.levels[self.level_idx].table;
                        while self.slot_idx < table.capacity() {
                            let idx = self.slot_idx;
                            self.slot_idx += 1;
                            if table.control_at(idx).is_occupied() {
                                let entry = unsafe { table.get_ref(idx) };
                                return Some((&entry.key, &entry.value));
                            }
                        }
                        self.level_idx += 1;
                        self.slot_idx = 0;
                    }
                    self.phase = FunnelIterPhase::Primary;
                    self.slot_idx = 0;
                }
                FunnelIterPhase::Primary => {
                    let table = &self.primary.table;
                    while self.slot_idx < table.capacity() {
                        let idx = self.slot_idx;
                        self.slot_idx += 1;
                        if table.control_at(idx).is_occupied() {
                            let entry = unsafe { table.get_ref(idx) };
                            return Some((&entry.key, &entry.value));
                        }
                    }
                    self.phase = FunnelIterPhase::Fallback;
                    self.slot_idx = 0;
                }
                FunnelIterPhase::Fallback => {
                    let table = &self.fallback.table;
                    while self.slot_idx < table.capacity() {
                        let idx = self.slot_idx;
                        self.slot_idx += 1;
                        if table.control_at(idx).is_occupied() {
                            let entry = unsafe { table.get_ref(idx) };
                            return Some((&entry.key, &entry.value));
                        }
                    }
                    self.phase = FunnelIterPhase::Done;
                }
                FunnelIterPhase::Done => return None,
            }
        }
    }
}

impl<'a, K, V> IntoIterator for &'a FunnelHashMap<K, V>
where
    K: Eq + Hash,
{
    type Item = (&'a K, &'a V);
    type IntoIter = FunnelIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Number of bucket levels for a given reserve fraction. Tighter reserve →
/// more levels (more probing budget per insert).
fn compute_level_count(reserve_fraction: f64) -> usize {
    ceil_to_usize((4.0 * (1.0 / reserve_fraction).log2() + 10.0).max(1.0))
}

/// Per-bucket slot count. Wider buckets reduce overflow into deeper levels
/// at the cost of more in-bucket probing.
fn compute_bucket_width(reserve_fraction: f64) -> usize {
    ceil_to_usize((2.0 * (1.0 / reserve_fraction).log2()).max(1.0))
}

/// Carve out the special-array capacity from the total. Returns
/// `(level_capacity, special_capacity)` such that levels get the bulk and
/// special gets a fraction sized to absorb expected overflow.
fn choose_special_capacity(
    total_capacity: usize,
    reserve_fraction: f64,
    bucket_size: usize,
) -> usize {
    if total_capacity == 0 {
        return 0;
    }

    let total_capacity_f64 = usize_to_f64(total_capacity);
    let lower_bound = ceil_to_usize((reserve_fraction * total_capacity_f64) / 2.0);
    let upper_bound = floor_to_usize((3.0 * reserve_fraction * total_capacity_f64) / 4.0);
    let lower_bound = lower_bound.min(total_capacity);
    let upper_bound = upper_bound.min(total_capacity);

    if lower_bound <= upper_bound {
        for special_capacity in (lower_bound..=upper_bound).rev() {
            if (total_capacity - special_capacity).is_multiple_of(bucket_size.max(1)) {
                return special_capacity;
            }
        }
    }

    let target = round_to_usize(
        ((5.0 * reserve_fraction * total_capacity_f64) / 8.0).clamp(0.0, total_capacity_f64),
    );

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
        let denom = 1.0 - ratio.powi(i32::try_from(level_count).expect("level count fits in i32"));
        if denom <= 0.0 {
            total_buckets.max(1)
        } else {
            round_to_usize((((usize_to_f64(total_buckets)) * (1.0 - ratio)) / denom).max(0.0))
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

        let mut chosen_bucket_count = None;
        let mut best_distance = usize::MAX;
        let candidate_upper_bound = max_next_bucket_count.min(remaining);
        for candidate_bucket_count in min_next_bucket_count..=candidate_upper_bound {
            let remaining_after_candidate = remaining - candidate_bucket_count;
            let (tail_min_sum, tail_max_sum) =
                possible_tail_sum_range(candidate_bucket_count, levels_after);
            if remaining_after_candidate < tail_min_sum || remaining_after_candidate > tail_max_sum
            {
                continue;
            }

            let distance = candidate_bucket_count.abs_diff(ideal_next_bucket_count);
            if distance < best_distance {
                best_distance = distance;
                chosen_bucket_count = Some(candidate_bucket_count);
                if distance == 0 {
                    break;
                }
            }
        }
        let chosen_bucket_count = chosen_bucket_count?;

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
    let mut min_sum = 0;
    let mut max_sum = 0;
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
        let max_insertions = max_insertions(capacity, DEFAULT_RESERVE_FRACTION);

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

    #[test]
    fn delete_heavy_preserves_correctness() {
        let mut map = FunnelHashMap::with_capacity(400);
        for i in 0..200 {
            map.insert(i, i * 10);
        }
        for i in 0..160 {
            assert_eq!(map.remove(&i), Some(i * 10));
        }
        for i in 160..200 {
            assert_eq!(
                map.get(&i),
                Some(&(i * 10)),
                "key {i} missing after deletes"
            );
        }
        assert_eq!(map.len(), 40);
        // Re-insert into tombstone-heavy map.
        for i in 1000..1100 {
            assert_eq!(map.insert(i, i), None);
        }
        for i in 1000..1100 {
            assert_eq!(map.get(&i), Some(&i), "key {i} missing after re-insert");
        }
    }

    #[test]
    fn large_map_correctness() {
        let n = 10_000;
        let mut map = FunnelHashMap::with_capacity(n * 2);
        for i in 0..n {
            assert_eq!(map.insert(i, i), None);
        }
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&i), "key {i} missing");
        }
        assert_eq!(map.len(), n);
    }

    #[test]
    fn interleaved_insert_delete_correctness() {
        let mut map = FunnelHashMap::with_capacity(256);
        // Insert 100, delete odd keys, verify even keys survive.
        for i in 0..100 {
            map.insert(i, i);
        }
        for i in (1..100).step_by(2) {
            assert!(map.remove(&i).is_some());
        }
        for i in (0..100).step_by(2) {
            assert_eq!(map.get(&i), Some(&i), "even key {i} missing");
        }
        for i in (1..100).step_by(2) {
            assert_eq!(map.get(&i), None, "odd key {i} should be gone");
        }
    }

    #[test]
    fn delete_insert_cycles_trigger_rebuild() {
        // Exercises the tombstone cleanup path: 6000 remove+insert cycles
        // on a 12K map forces level.tombstones > capacity/2.
        let n = 12_000;
        let mut map = FunnelHashMap::with_capacity(n * 2);
        for i in 0..n {
            map.insert(i, i);
        }

        for i in 0..6000 {
            assert!(map.remove(&i).is_some(), "remove {i} failed");
            map.insert(i + n, i + n);
        }

        assert_eq!(map.len(), n);
        // Verify all remaining keys are findable.
        for i in 6000..n {
            assert_eq!(map.get(&i), Some(&i), "original key {i} missing");
        }
        for i in 0..6000 {
            assert_eq!(
                map.get(&(i + n)),
                Some(&(i + n)),
                "new key {} missing",
                i + n
            );
        }
    }

    #[test]
    fn iter_yields_every_inserted_pair_once() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(128);
        for i in 0..80 {
            map.insert(i, i * 7);
        }
        let mut collected: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        collected.sort();
        let expected: Vec<(i32, i32)> = (0..80).map(|i| (i, i * 7)).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn iter_skips_tombstones_after_remove() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(64);
        for i in 0..40 {
            map.insert(i, i);
        }
        for i in (0..40).step_by(3) {
            map.remove(&i);
        }
        let keys: Vec<i32> = map.iter().map(|(&k, _)| k).collect();
        assert_eq!(keys.len(), map.len());
    }

    #[test]
    fn iter_empty_map_is_empty() {
        let map: FunnelHashMap<i32, i32> = FunnelHashMap::new();
        assert_eq!(map.iter().count(), 0);
    }

    #[test]
    fn multi_get_matches_get_for_hits_and_misses() {
        let n: i32 = 1_000;
        let cap = usize::try_from(n * 2).expect("positive capacity");
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(cap);
        for i in 0..n {
            map.insert(i, i * 7);
        }

        let probe_keys: Vec<i32> = (-100..(n + 100)).collect();
        let refs: Vec<&i32> = probe_keys.iter().collect();
        let batched = map.multi_get(&refs);
        assert_eq!(batched.len(), refs.len());
        for (k, got) in refs.iter().zip(batched.iter()) {
            let expected = map.get(*k);
            assert_eq!(got.copied(), expected.copied(), "mismatch on key {k}");
        }
    }

    #[test]
    fn multi_get_on_empty_map_returns_all_none() {
        let map: FunnelHashMap<i32, i32> = FunnelHashMap::new();
        let keys = [1, 2, 3];
        let refs: Vec<&i32> = keys.iter().collect();
        let out = map.multi_get(&refs);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(Option::is_none));
    }

    #[test]
    fn get_disjoint_mut_returns_all_refs_on_hits() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(64);
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
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(32);
        for i in 0..8 {
            map.insert(i, i);
        }

        assert!(map.get_disjoint_mut([&0, &1, &99]).is_none());
    }

    #[test]
    #[should_panic(expected = "duplicate keys")]
    fn get_disjoint_mut_panics_on_duplicate_keys() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(32);
        map.insert(1, 100);
        map.insert(2, 200);
        let _ = map.get_disjoint_mut([&1, &1]);
    }

    #[test]
    fn get_disjoint_mut_zero_keys_is_some_empty() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(16);
        map.insert(1, 1);
        let got: [&mut i32; 0] = map
            .get_disjoint_mut::<i32, 0>([])
            .expect("zero-key returns Some");
        assert_eq!(got.len(), 0);
    }

    #[test]
    fn get_disjoint_mut_mutation_propagates() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(32);
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
    fn multi_get_at_pipeline_depth_boundary() {
        let n: i32 = 32;
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(64);
        for i in 0..n {
            map.insert(i, i + 1);
        }

        for batch_size in [7usize, 8, 9, 16, 17] {
            let probe_keys: Vec<i32> = (0..i32::try_from(batch_size).expect("fits")).collect();
            let refs: Vec<&i32> = probe_keys.iter().collect();
            let batched = map.multi_get(&refs);
            assert_eq!(batched.len(), batch_size, "len at N={batch_size}");
            for (k, got) in refs.iter().zip(batched.iter()) {
                assert_eq!(got.copied(), map.get(*k).copied(), "N={batch_size} key={k}");
            }
        }
    }

    #[test]
    fn multi_get_miss_heavy_batch() {
        let n: i32 = 1_000;
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(2_000);
        for i in 0..n {
            map.insert(i, i);
        }

        let miss_keys: Vec<i32> = ((n + 1_000)..(n + 2_000)).collect();
        let refs: Vec<&i32> = miss_keys.iter().collect();
        let out = map.multi_get(&refs);
        assert_eq!(out.len(), miss_keys.len());
        assert!(
            out.iter().all(Option::is_none),
            "all-miss batch should return all None"
        );
    }

    #[test]
    fn multi_get_duplicate_keys_in_batch_yield_same_value() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(32);
        map.insert(7, 700);
        map.insert(13, 1300);

        let keys = [&7, &13, &7, &13, &7];
        let out = map.multi_get(&keys);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].copied(), Some(700));
        assert_eq!(out[1].copied(), Some(1300));
        assert_eq!(out[2].copied(), Some(700));
        assert_eq!(out[3].copied(), Some(1300));
        assert_eq!(out[4].copied(), Some(700));
    }

    #[test]
    fn multi_get_into_reuses_buffer_across_calls() {
        let mut map: FunnelHashMap<i32, i32> = FunnelHashMap::with_capacity(64);
        for i in 0..16 {
            map.insert(i, i * 2);
        }

        let mut out: Vec<Option<&i32>> = Vec::with_capacity(32);
        let keys1: Vec<i32> = (0..8).collect();
        let refs1: Vec<&i32> = keys1.iter().collect();
        map.multi_get_into(&refs1, &mut out);
        assert_eq!(out.len(), 8);
        for (k, v) in refs1.iter().zip(out.iter()) {
            assert_eq!(v.copied(), Some(*k * 2));
        }

        let keys2: Vec<i32> = (8..16).collect();
        let refs2: Vec<&i32> = keys2.iter().collect();
        map.multi_get_into(&refs2, &mut out);
        assert_eq!(out.len(), 8);
        for (k, v) in refs2.iter().zip(out.iter()) {
            assert_eq!(v.copied(), Some(*k * 2));
        }

        map.multi_get_into::<i32>(&[], &mut out);
        assert!(out.is_empty());
    }
}

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

use super::bitmask::BitMask;
use super::simd::{eq_mask_16, free_mask_16};

use super::math::round_up_to_group;

pub(crate) const GROUP_SIZE: usize = 16;

/// Alignment for the control-byte region. Matches 64-byte cache lines so
/// the first group is line-aligned and groups pack 4-per-line without splits.
const CONTROL_ALIGN: usize = 64;

// ---------------------------------------------------------------------------
// Mini-hash sidecar — feature-gated
// ---------------------------------------------------------------------------
//
// When `mini-hash` is enabled, every slot carries a 32-bit secondary hash
// derived from the same full hash that produced its 7-bit control
// fingerprint. After a SIMD fingerprint match, the mini-hash is consulted
// before any `K::eq` call. This drops the fingerprint false-positive rate
// from ~1/128 (7 bits) to ~1/2^32, which matters when `K::eq` is expensive
// (Python `HashedAny::eq` crosses the GIL; an `Arc<str>` key chases two
// pointers; etc.).
//
// When the feature is off, the sidecar slab is omitted from the layout (no
// 4 B/slot memory tax), `MiniHash` collapses to `()`, and the gate is
// erased from the lookup hot path via `mini_matches` returning `true`
// unconditionally.

/// Per-slot mini-hash type. `u32` when the `mini-hash` feature is on,
/// otherwise a zero-size marker so callers don't need to `#[cfg]` every
/// site that threads it through.
#[cfg(feature = "mini-hash")]
pub(crate) type MiniHash = u32;
#[cfg(not(feature = "mini-hash"))]
pub(crate) type MiniHash = ();

/// Derive the per-slot mini-hash from a full 64-bit key hash.
///
/// We take the low 32 bits. The 7-bit control fingerprint is the *top* 7
/// bits (`hash >> 57`), so the two metadata channels are extracted from
/// disjoint bit positions of the same 64-bit word. Independence between
/// fingerprint and mini-hash therefore relies on the hasher's avalanche
/// (every output bit depends on every input bit) rather than a structural
/// argument: a hasher that concentrates entropy in the high bits would
/// make the mini-hash low-quality. With `foldhash` (the default) this is
/// fine; callers swapping in a custom `BuildHasher` should ensure it
/// avalanches.
#[cfg(feature = "mini-hash")]
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn mini_hash(hash: u64) -> MiniHash {
    hash as u32
}

#[cfg(not(feature = "mini-hash"))]
#[inline]
#[must_use]
#[allow(clippy::missing_const_for_fn)]
pub(crate) fn mini_hash(_hash: u64) -> MiniHash {}

pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

/// A flat hash table: one allocation holds slots, control bytes, and (when
/// the `mini-hash` feature is enabled) a per-slot 32-bit mini-hash sidecar.
///
/// ```text
/// mini-hash on:  [slots: capacity * sizeof(T)] [pad] [controls: group_count * 16 (64-aligned)] [pad] [mini-hashes: group_count * 16 * u32 (64-aligned)]
/// mini-hash off: [slots: capacity * sizeof(T)] [pad] [controls: group_count * 16 (64-aligned)]
/// ```
///
/// `data_ptr` points to the start of the slots array; control bytes and
/// mini-hashes live at fixed offsets, accessed via `ctrl_ptr()` /
/// `mini_ptr()`. The mini-hash sidecar (when present) carries one `u32`
/// per slot and is consulted after the SIMD fingerprint match, before any
/// key compare. This drops the false-positive rate of the 7-bit
/// fingerprint (~1/128) down to ~1/2^32, which matters when key equality
/// is expensive (e.g. crossing the GIL for `HashedAny::eq`).
///
/// Note: the mini-hash for unoccupied slots (FREE or TOMBSTONE) is inert
/// because the control-byte SIMD scan never matches those bytes, so callers
/// never reach the mini-hash gate on them. Stale values left behind by
/// `mark_tombstone` are therefore harmless; `mini_at` `debug_assert!`s the
/// slot is occupied to catch any future caller that tries to short-circuit
/// tombstone reuse on the sidecar.
pub(crate) struct RawTable<T> {
    data_ptr: NonNull<u8>,
    capacity: usize,
    group_count: usize,
    ctrl_offset: usize,
    #[cfg(feature = "mini-hash")]
    mini_offset: usize,
    _marker: PhantomData<T>,
}

// SAFETY: RawTable<T> owns its allocation exclusively; data_ptr is not aliased.
// Sending across threads is sound when T: Send (same as Box<[T]>). Sync requires
// T: Sync because shared &RawTable<T> can hand out shared &T via get_ref.
unsafe impl<T: Send> Send for RawTable<T> {}
unsafe impl<T: Sync> Sync for RawTable<T> {}

impl<T> std::fmt::Debug for RawTable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTable")
            .field("capacity", &self.capacity)
            .field("group_count", &self.group_count)
            .finish_non_exhaustive()
    }
}

impl<T> Drop for RawTable<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Self::unified_layout(self.capacity, self.group_count).0;
            unsafe { alloc::dealloc(self.data_ptr.as_ptr(), layout) };
        }
    }
}

impl<T> RawTable<T> {
    pub fn new(capacity: usize) -> Self {
        // ZSTs would make `Layout::array::<T>(capacity)` a no-op and let
        // `capacity` (and thus `group_count`) grow without bound, which would
        // then overflow the control / mini-hash sidecar layout math below.
        // RawTable is only instantiated with `Entry<K, V>` and `u64` today; the
        // assertion is constant-folded for those types and pins the invariant
        // for any future callers.
        const {
            assert!(
                std::mem::size_of::<T>() != 0,
                "RawTable<T> requires a non-ZST T"
            );
        };

        if capacity == 0 {
            return Self {
                data_ptr: NonNull::dangling(),
                capacity: 0,
                group_count: 0,
                ctrl_offset: 0,
                #[cfg(feature = "mini-hash")]
                mini_offset: 0,
                _marker: PhantomData,
            };
        }

        let padded_capacity = round_up_to_group(capacity);
        let group_count = padded_capacity / GROUP_SIZE;
        let layout_info = Self::unified_layout(capacity, group_count);
        let layout = layout_info.0;
        let ctrl_offset = layout_info.1;
        #[cfg(feature = "mini-hash")]
        let mini_offset = layout_info.2;

        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let data_ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        Self {
            data_ptr,
            capacity,
            group_count,
            ctrl_offset,
            #[cfg(feature = "mini-hash")]
            mini_offset,
            _marker: PhantomData,
        }
    }

    /// Layout:
    /// - mini-hash on: `[slots] [pad] [controls (64-aligned)] [pad] [mini (64-aligned)]`
    /// - mini-hash off: `[slots] [pad] [controls (64-aligned)]`
    ///
    /// Returns `(layout, ctrl_offset, mini_offset)`. `mini_offset` is `0`
    /// when the feature is off.
    fn unified_layout(capacity: usize, group_count: usize) -> (Layout, usize, usize) {
        let slots_layout = Layout::array::<T>(capacity).expect("slots layout overflow");
        let controls_size = group_count
            .checked_mul(GROUP_SIZE)
            .expect("controls layout overflow");
        let controls_layout = Layout::from_size_align(controls_size, CONTROL_ALIGN)
            .expect("controls layout overflow");
        let (combined, ctrl_offset) = slots_layout
            .extend(controls_layout)
            .expect("layout extend overflow");

        #[cfg(feature = "mini-hash")]
        {
            let mini_size = controls_size
                .checked_mul(std::mem::size_of::<u32>())
                .expect("mini-hash layout overflow");
            let mini_layout = Layout::from_size_align(mini_size, CONTROL_ALIGN)
                .expect("mini-hash layout overflow");
            let (combined, mini_offset) = combined
                .extend(mini_layout)
                .expect("layout extend overflow");
            (combined.pad_to_align(), ctrl_offset, mini_offset)
        }
        #[cfg(not(feature = "mini-hash"))]
        {
            (combined.pad_to_align(), ctrl_offset, 0)
        }
    }

    #[inline]
    fn slots_ptr(&self) -> *mut T {
        self.data_ptr.as_ptr().cast::<T>()
    }

    #[inline]
    fn ctrl_ptr(&self) -> *mut u8 {
        unsafe { self.data_ptr.as_ptr().add(self.ctrl_offset) }
    }

    #[cfg(feature = "mini-hash")]
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    fn mini_ptr(&self) -> *mut u32 {
        // mini_offset is aligned to CONTROL_ALIGN (64), so the cast to *mut u32
        // is sound.
        unsafe { self.data_ptr.as_ptr().add(self.mini_offset).cast::<u32>() }
    }

    /// Per-slot 32-bit secondary hash used to short-circuit key compares
    /// after a fingerprint match.
    ///
    /// Panics in debug builds if `idx` does not refer to an OCCUPIED slot
    /// (free / tombstone slots may carry stale mini-hash values left by a
    /// prior occupant). In release builds the read is safe because callers
    /// always reach this function via a SIMD fingerprint match on the
    /// control byte, which is mutually exclusive with FREE / TOMBSTONE.
    #[cfg(feature = "mini-hash")]
    #[inline]
    pub(crate) fn mini_at(&self, idx: usize) -> u32 {
        debug_assert!(
            super::control::ControlByte::is_occupied(&self.control_at(idx)),
            "mini_at called on non-occupied slot {idx} (stale mini-hash)"
        );
        unsafe { *self.mini_ptr().add(idx) }
    }

    /// Write the per-slot mini-hash. `pub(crate)` so it stays an
    /// implementation detail of the table — callers go through
    /// `write_with_control`.
    #[cfg(feature = "mini-hash")]
    #[inline]
    pub(crate) fn set_mini_hash(&mut self, idx: usize, mini: u32) {
        unsafe { *self.mini_ptr().add(idx) = mini };
    }

    /// Compare a candidate mini-hash against the slot's stored mini-hash.
    /// Returns `true` unconditionally when the `mini-hash` feature is off
    /// so the gate compiles out of the lookup hot path.
    #[cfg(feature = "mini-hash")]
    #[inline]
    pub(crate) fn mini_matches(&self, idx: usize, key_mini: MiniHash) -> bool {
        self.mini_at(idx) == key_mini
    }

    #[cfg(not(feature = "mini-hash"))]
    #[inline]
    #[allow(clippy::unused_self, clippy::missing_const_for_fn)]
    pub(crate) fn mini_matches(&self, _idx: usize, _key_mini: MiniHash) -> bool {
        true
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    #[inline]
    pub fn group_data_ptr(&self, group_idx: usize) -> *const u8 {
        unsafe { self.ctrl_ptr().add(group_idx * GROUP_SIZE) }
    }

    #[inline]
    pub fn control_at(&self, idx: usize) -> u8 {
        unsafe { *self.ctrl_ptr().add(idx) }
    }

    #[inline]
    pub fn write(&mut self, idx: usize, value: T) {
        unsafe { self.slots_ptr().add(idx).write(value) };
    }

    /// Write entry, control byte, and (when the feature is on) the mini-hash.
    /// The `mini` argument is `()` when off — a zero-cost marker so callers
    /// don't need to `#[cfg]` each site.
    #[inline]
    pub fn write_with_control(&mut self, idx: usize, value: T, control: u8, mini: MiniHash) {
        self.write(idx, value);
        self.set_control(idx, control);
        #[cfg(feature = "mini-hash")]
        {
            self.set_mini_hash(idx, mini);
        }
        #[cfg(not(feature = "mini-hash"))]
        {
            let () = mini;
        }
    }

    #[inline]
    pub fn set_control(&mut self, idx: usize, new_control: u8) {
        unsafe { *self.ctrl_ptr().add(idx) = new_control };
    }

    #[inline]
    pub fn mark_tombstone(&mut self, idx: usize) {
        // Note: the stored mini-hash is left stale. Inert because the
        // control-byte SIMD scan never matches tombstones, so the gate is
        // unreachable for this slot until it's reused.
        self.set_control(idx, super::control::CTRL_TOMBSTONE);
    }

    #[inline]
    pub fn clear_all_controls(&mut self) {
        if self.group_count == 0 {
            return;
        }
        unsafe {
            ptr::write_bytes(self.ctrl_ptr(), 0, self.group_count * GROUP_SIZE);
        }
    }

    #[inline]
    pub unsafe fn get_ref(&self, idx: usize) -> &T {
        unsafe { &*self.slots_ptr().add(idx) }
    }

    #[inline]
    pub unsafe fn get_mut(&mut self, idx: usize) -> &mut T {
        unsafe { &mut *self.slots_ptr().add(idx) }
    }

    #[inline]
    pub unsafe fn take(&mut self, idx: usize) -> T {
        unsafe { self.slots_ptr().add(idx).read() }
    }

    #[inline]
    pub unsafe fn drop_in_place(&mut self, idx: usize) {
        unsafe { ptr::drop_in_place(self.slots_ptr().add(idx)) }
    }

    #[inline]
    pub fn group_match_mask(&self, group_idx: usize, target: u8) -> BitMask {
        let ptr = unsafe { self.ctrl_ptr().add(group_idx * GROUP_SIZE) };
        unsafe { eq_mask_16(ptr, target) }
    }

    #[inline]
    pub fn group_free_mask(&self, group_idx: usize) -> BitMask {
        let ptr = unsafe { self.ctrl_ptr().add(group_idx * GROUP_SIZE) };
        unsafe { free_mask_16(ptr) }
    }

    #[inline]
    pub fn first_free_in_group(&self, group_idx: usize) -> Option<usize> {
        let offset = self.group_free_mask(group_idx).lowest()?;
        let slot_idx = group_idx * GROUP_SIZE + offset;
        if slot_idx < self.capacity {
            Some(slot_idx)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout invariant: under both feature configurations the control bytes
    /// follow the slots region with 64-byte alignment, and (when the
    /// mini-hash feature is on) the mini-hash slab follows the controls
    /// with 64-byte alignment and is sized to exactly one u32 per slot.
    #[test]
    fn unified_layout_offsets_are_well_aligned() {
        // Spot-check several capacities, including a partial last group.
        for cap in [16usize, 17, 64, 128, 257, 1024] {
            let group_count = round_up_to_group(cap) / GROUP_SIZE;
            let (layout, ctrl_offset, mini_offset) =
                RawTable::<u64>::unified_layout(cap, group_count);
            assert!(ctrl_offset.is_multiple_of(CONTROL_ALIGN));
            assert!(layout.size() >= ctrl_offset + group_count * GROUP_SIZE);
            #[cfg(feature = "mini-hash")]
            {
                assert!(mini_offset.is_multiple_of(CONTROL_ALIGN));
                let mini_size = group_count * GROUP_SIZE * std::mem::size_of::<u32>();
                assert!(layout.size() >= mini_offset + mini_size);
            }
            #[cfg(not(feature = "mini-hash"))]
            {
                // mini_offset is unused when the feature is off; the
                // unused binding is materialized by the return tuple.
                assert_eq!(mini_offset, 0);
            }
        }
    }

    /// `mini_hash` extracts the low 32 bits; the 7-bit control fingerprint
    /// extracts `hash >> 57`. The two are bit-disjoint by construction so a
    /// hasher with full avalanche makes the channels independent.
    #[cfg(feature = "mini-hash")]
    #[test]
    fn mini_hash_and_fingerprint_extract_disjoint_bits() {
        use super::super::simd::ControlOps;
        // Construct two distinct full hashes that share the top 7 bits but
        // differ in the low 32. Their mini-hashes must differ; their
        // fingerprints must match.
        let h_a: u64 = 0xCAFEBABE_DEADBEEFu64;
        let h_b: u64 = (h_a & 0xFE00_0000_0000_0000) | 0x0000_0000_1234_5678;
        assert_eq!(
            ControlOps::control_fingerprint(h_a),
            ControlOps::control_fingerprint(h_b),
            "constructed hashes must share fingerprint",
        );
        assert_ne!(mini_hash(h_a), mini_hash(h_b));
    }
}

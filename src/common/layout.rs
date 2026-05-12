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

/// Derive the 32-bit sidecar value from a full 64-bit hash. Uses bits
/// disjoint from the 7-bit control fingerprint (`hash >> 57`) so the two
/// metadata channels collide independently.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn mini_hash_from_full(hash: u64) -> u32 {
    // Take the low 32 bits — fully disjoint from the fingerprint's top 7 bits.
    hash as u32
}

pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

/// A flat hash table: one allocation holds slots, control bytes, and a
/// per-slot 32-bit mini-hash sidecar.
///
/// ```text
/// [slots: capacity * sizeof(T)] [pad] [controls: group_count * 16 (64-aligned)] [pad] [mini_hashes: group_count * 16 * u32 (64-aligned)]
/// ```
///
/// `data_ptr` points to the start of the slots array; control bytes and
/// mini-hashes live at fixed offsets, accessed via `ctrl_ptr()` /
/// `mini_ptr()`. The mini-hash sidecar carries one `u32` per slot and is
/// consulted after the SIMD fingerprint match, before any key compare. This
/// drops the false-positive rate of the 7-bit fingerprint (~1/128) down to
/// ~1/2^32, which matters when key equality is expensive (e.g. crossing the
/// GIL for `HashedAny::eq`).
pub(crate) struct RawTable<T> {
    data_ptr: NonNull<u8>,
    capacity: usize,
    group_count: usize,
    ctrl_offset: usize,
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
            let (layout, _, _) = Self::unified_layout(self.capacity, self.group_count);
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
                mini_offset: 0,
                _marker: PhantomData,
            };
        }

        let padded_capacity = round_up_to_group(capacity);
        let group_count = padded_capacity / GROUP_SIZE;
        let (layout, ctrl_offset, mini_offset) = Self::unified_layout(capacity, group_count);

        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let data_ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        Self {
            data_ptr,
            capacity,
            group_count,
            ctrl_offset,
            mini_offset,
            _marker: PhantomData,
        }
    }

    /// Layout: `[slots (T-aligned)] [pad] [controls (64-aligned)] [pad] [mini-hashes (64-aligned)]`.
    fn unified_layout(capacity: usize, group_count: usize) -> (Layout, usize, usize) {
        let slots_layout = Layout::array::<T>(capacity).expect("slots layout overflow");
        let controls_size = group_count
            .checked_mul(GROUP_SIZE)
            .expect("controls layout overflow");
        let controls_layout = Layout::from_size_align(controls_size, CONTROL_ALIGN)
            .expect("controls layout overflow");
        let mini_size = controls_size
            .checked_mul(std::mem::size_of::<u32>())
            .expect("mini-hash layout overflow");
        let mini_layout =
            Layout::from_size_align(mini_size, CONTROL_ALIGN).expect("mini-hash layout overflow");
        let (combined, ctrl_offset) = slots_layout
            .extend(controls_layout)
            .expect("layout extend overflow");
        let (combined, mini_offset) = combined
            .extend(mini_layout)
            .expect("layout extend overflow");
        (combined.pad_to_align(), ctrl_offset, mini_offset)
    }

    #[inline]
    fn slots_ptr(&self) -> *mut T {
        self.data_ptr.as_ptr().cast::<T>()
    }

    #[inline]
    fn ctrl_ptr(&self) -> *mut u8 {
        unsafe { self.data_ptr.as_ptr().add(self.ctrl_offset) }
    }

    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    fn mini_ptr(&self) -> *mut u32 {
        // mini_offset is aligned to CONTROL_ALIGN (64), so the cast to *mut u32
        // is sound.
        unsafe { self.data_ptr.as_ptr().add(self.mini_offset).cast::<u32>() }
    }

    /// Per-slot 32-bit secondary hash used to short-circuit key compares after
    /// a fingerprint match.
    #[inline]
    pub fn mini_hash_at(&self, idx: usize) -> u32 {
        unsafe { *self.mini_ptr().add(idx) }
    }

    #[inline]
    pub fn set_mini_hash(&mut self, idx: usize, mini: u32) {
        unsafe { *self.mini_ptr().add(idx) = mini };
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

    #[inline]
    pub fn write_with_control(&mut self, idx: usize, value: T, control: u8, mini: u32) {
        self.write(idx, value);
        self.set_control(idx, control);
        self.set_mini_hash(idx, mini);
    }

    #[inline]
    pub fn set_control(&mut self, idx: usize, new_control: u8) {
        unsafe { *self.ctrl_ptr().add(idx) = new_control };
    }

    #[inline]
    pub fn mark_tombstone(&mut self, idx: usize) {
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

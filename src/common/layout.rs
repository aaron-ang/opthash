use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

use super::bitmask::BitMask;
use super::simd::{eq_mask_16, free_mask_16, match_and_free_masks_16};

use super::math::round_up_to_group;

pub(crate) const GROUP_SIZE: usize = 16;

/// Alignment for the control-byte region. Matches 64-byte cache lines so
/// the first group is line-aligned and groups pack 4-per-line without splits.
const CONTROL_ALIGN: usize = 64;

#[derive(Debug)]
pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

/// A flat hash table: one allocation holds slots then control bytes.
///
/// ```text
/// [slots: capacity * sizeof(T)] [padding for 16-byte alignment] [controls: group_count * 16]
/// ```
///
/// `data_ptr` points to the start of the slots array. Control bytes live at
/// a fixed offset after the slots, accessed via `ctrl_ptr()`.
pub(crate) struct RawTable<T> {
    data_ptr: NonNull<u8>,
    capacity: usize,
    group_count: usize,
    ctrl_offset: usize,
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
            let (layout, _) = Self::unified_layout(self.capacity, self.group_count);
            unsafe { alloc::dealloc(self.data_ptr.as_ptr(), layout) };
        }
    }
}

impl<T> RawTable<T> {
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                data_ptr: NonNull::dangling(),
                capacity: 0,
                group_count: 0,
                ctrl_offset: 0,
                _marker: PhantomData,
            };
        }

        let padded_capacity = round_up_to_group(capacity);
        let group_count = padded_capacity / GROUP_SIZE;
        let (layout, ctrl_offset) = Self::unified_layout(capacity, group_count);

        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let data_ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        Self {
            data_ptr,
            capacity,
            group_count,
            ctrl_offset,
            _marker: PhantomData,
        }
    }

    /// Layout: `[slots (T-aligned)] [padding] [controls (64-aligned)]`.
    fn unified_layout(capacity: usize, group_count: usize) -> (Layout, usize) {
        let slots_layout = Layout::array::<T>(capacity).expect("slots layout overflow");
        let controls_layout = Layout::from_size_align(group_count * GROUP_SIZE, CONTROL_ALIGN)
            .expect("controls layout overflow");
        let (combined, ctrl_offset) = slots_layout
            .extend(controls_layout)
            .expect("layout extend overflow");
        (combined.pad_to_align(), ctrl_offset)
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
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// Returns a raw pointer to the control bytes of group `group_idx`.
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
    pub fn write_with_control(&mut self, idx: usize, value: T, control: u8) {
        self.write(idx, value);
        self.set_control(idx, control);
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
    pub fn group_match_and_free_mask(&self, group_idx: usize, target: u8) -> (BitMask, BitMask) {
        let ptr = unsafe { self.ctrl_ptr().add(group_idx * GROUP_SIZE) };
        unsafe { match_and_free_masks_16(ptr, target) }
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

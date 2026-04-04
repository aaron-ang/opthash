use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

use opthash_internal::{eq_mask_16, free_mask_16};

use super::control::{CTRL_EMPTY, CTRL_TOMBSTONE, ControlByte, valid_group_mask};
use super::math::round_up_to_group;

pub(crate) const GROUP_SIZE: usize = 16;

/// Compact stride: control bytes only, no embedded metadata.
pub(crate) const COMPACT_STRIDE: usize = GROUP_SIZE;

/// Wide stride: 32-byte blocks with embedded `GroupMeta` after the control bytes.
///   bytes  0-15: control bytes (fingerprints / sentinels)
///   byte  16:    live count
///   byte  17:    tombstone count
///   byte  18:    full flag (0 = not full, 1 = full)
///   bytes 19-31: padding
pub(crate) const META_STRIDE: usize = 32;

const META_LIVE_OFFSET: usize = 16;
const META_TOMBSTONES_OFFSET: usize = 17;
const META_FULL_OFFSET: usize = 18;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GroupMeta {
    pub(crate) live: u8,
    pub(crate) tombstones: u8,
    pub(crate) full: bool,
}

#[derive(Debug)]
pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

/// A raw table of slots with interleaved control bytes.
///
/// The `STRIDE` const generic controls the per-group block size:
/// - `COMPACT_STRIDE` (16): flat control-byte array, no embedded metadata.
/// - `META_STRIDE` (32): each group occupies a 32-byte block with controls + `GroupMeta`.
pub(crate) struct RawTable<T, const STRIDE: usize> {
    slots_ptr: NonNull<T>,
    data_ptr: NonNull<u8>,
    capacity: usize,
    group_count: usize,
    _marker: PhantomData<T>,
}

impl<T, const STRIDE: usize> std::fmt::Debug for RawTable<T, STRIDE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTable")
            .field("capacity", &self.capacity)
            .field("group_count", &self.group_count)
            .field("stride", &STRIDE)
            .finish_non_exhaustive()
    }
}

impl<T, const STRIDE: usize> Drop for RawTable<T, STRIDE> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                alloc::dealloc(
                    self.slots_ptr.as_ptr().cast::<u8>(),
                    Self::slots_layout(self.capacity),
                );
            };
        }
        if self.group_count > 0 {
            unsafe {
                alloc::dealloc(self.data_ptr.as_ptr(), Self::data_layout(self.group_count));
            };
        }
    }
}

impl<T, const STRIDE: usize> RawTable<T, STRIDE> {
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                slots_ptr: NonNull::dangling(),
                data_ptr: NonNull::dangling(),
                capacity: 0,
                group_count: 0,
                _marker: PhantomData,
            };
        }

        let padded_capacity = round_up_to_group(capacity);
        let group_count = padded_capacity / GROUP_SIZE;

        let slots_layout = Self::slots_layout(capacity);
        let raw_slots = unsafe { alloc::alloc(slots_layout) };
        let slots_ptr = NonNull::new(raw_slots.cast::<T>())
            .unwrap_or_else(|| alloc::handle_alloc_error(slots_layout));

        let data_layout = Self::data_layout(group_count);
        let raw_data = unsafe { alloc::alloc_zeroed(data_layout) };
        let data_ptr =
            NonNull::new(raw_data).unwrap_or_else(|| alloc::handle_alloc_error(data_layout));

        Self {
            slots_ptr,
            data_ptr,
            capacity,
            group_count,
            _marker: PhantomData,
        }
    }

    fn slots_layout(capacity: usize) -> Layout {
        Layout::array::<T>(capacity).expect("slot layout overflow")
    }

    fn data_layout(group_count: usize) -> Layout {
        let align = STRIDE.max(GROUP_SIZE);
        Layout::from_size_align(group_count * STRIDE, align).expect("data block layout overflow")
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
    pub fn group_start(group_idx: usize) -> usize {
        group_idx * GROUP_SIZE
    }

    #[inline]
    pub fn group_len(&self, group_idx: usize) -> usize {
        logical_group_len(self.capacity, group_idx)
    }

    #[inline]
    fn data_offset(group_idx: usize, slot_offset: usize) -> usize {
        group_idx * STRIDE + slot_offset
    }

    /// Returns a slice of the 16 control bytes for the given group.
    #[inline]
    pub fn group_controls(&self, group_idx: usize) -> &[u8] {
        debug_assert!(group_idx < self.group_count);
        unsafe {
            std::slice::from_raw_parts(
                self.data_ptr.as_ptr().add(Self::data_offset(group_idx, 0)),
                GROUP_SIZE,
            )
        }
    }

    /// Returns a slice of `len` contiguous control bytes starting at `start_slot`.
    ///
    /// `start_slot` and `start_slot + len` must fall within the same group block
    /// (i.e., `len <= GROUP_SIZE` and the slot range must not cross a group boundary).
    #[inline]
    pub fn bucket_controls(&self, start_slot: usize, len: usize) -> &[u8] {
        let group_idx = start_slot / GROUP_SIZE;
        let offset = start_slot % GROUP_SIZE;
        debug_assert!(group_idx < self.group_count);
        debug_assert!(offset + len <= GROUP_SIZE, "bucket spans a group boundary");
        unsafe {
            std::slice::from_raw_parts(
                self.data_ptr
                    .as_ptr()
                    .add(Self::data_offset(group_idx, offset)),
                len,
            )
        }
    }

    #[inline]
    pub fn control_at(&self, idx: usize) -> u8 {
        let group_idx = idx / GROUP_SIZE;
        let slot_offset = idx % GROUP_SIZE;
        unsafe {
            *self
                .data_ptr
                .as_ptr()
                .add(Self::data_offset(group_idx, slot_offset))
        }
    }

    #[inline]
    pub fn write(&mut self, idx: usize, value: T) {
        unsafe { self.slots_ptr.as_ptr().add(idx).write(value) };
    }

    #[inline]
    pub fn write_with_control(&mut self, idx: usize, value: T, control: u8) {
        self.write(idx, value);
        self.set_control(idx, control);
    }

    #[inline]
    pub fn set_control(&mut self, idx: usize, new_control: u8) {
        let old_control = self.control_at(idx);
        if old_control == new_control {
            return;
        }

        let group_idx = idx / GROUP_SIZE;
        let slot_offset = idx % GROUP_SIZE;
        let base = unsafe { self.data_ptr.as_ptr().add(Self::data_offset(group_idx, 0)) };
        unsafe { *base.add(slot_offset) = new_control };
        self.adjust_group_meta_if_wide(base, old_control, new_control, group_idx);
    }

    #[inline]
    pub fn clear_control(&mut self, idx: usize) {
        self.set_control(idx, CTRL_EMPTY);
    }

    #[inline]
    pub fn mark_tombstone(&mut self, idx: usize) {
        self.set_control(idx, CTRL_TOMBSTONE);
    }

    #[inline]
    pub fn clear_all_controls(&mut self) {
        if self.group_count == 0 {
            return;
        }
        unsafe {
            ptr::write_bytes(self.data_ptr.as_ptr(), 0, self.group_count * STRIDE);
        }
    }

    #[inline]
    pub unsafe fn get_ref(&self, idx: usize) -> &T {
        unsafe { &*self.slots_ptr.as_ptr().add(idx) }
    }

    #[inline]
    pub unsafe fn get_mut(&mut self, idx: usize) -> &mut T {
        unsafe { &mut *self.slots_ptr.as_ptr().add(idx) }
    }

    #[inline]
    pub unsafe fn take(&mut self, idx: usize) -> T {
        unsafe { self.slots_ptr.as_ptr().add(idx).read() }
    }

    #[inline]
    pub unsafe fn drop_in_place(&mut self, idx: usize) {
        unsafe { ptr::drop_in_place(self.slots_ptr.as_ptr().add(idx)) }
    }

    #[inline]
    pub fn group_match_mask(&self, group_idx: usize, target: u8) -> u16 {
        let valid = valid_group_mask(self.group_len(group_idx));
        let ptr = unsafe { self.data_ptr.as_ptr().add(Self::data_offset(group_idx, 0)) };
        let mask = unsafe { eq_mask_16(ptr, target) };
        mask & valid
    }

    #[inline]
    pub fn group_free_mask(&self, group_idx: usize) -> u16 {
        let valid = valid_group_mask(self.group_len(group_idx));
        let ptr = unsafe { self.data_ptr.as_ptr().add(Self::data_offset(group_idx, 0)) };
        let mask = unsafe { free_mask_16(ptr) };
        mask & valid
    }

    #[inline]
    pub fn first_free_in_group(&self, group_idx: usize, start_offset: usize) -> Option<usize> {
        let start_mask = offset_mask(start_offset);
        let mask = self.group_free_mask(group_idx) & start_mask;
        if mask == 0 {
            None
        } else {
            Some(Self::group_start(group_idx) + mask.trailing_zeros() as usize)
        }
    }

    /// Updates `GroupMeta` bytes embedded in the block — only when `STRIDE >= META_STRIDE`.
    /// For compact tables (`STRIDE == COMPACT_STRIDE`), this is a no-op that the compiler
    /// eliminates entirely.
    #[inline]
    fn adjust_group_meta_if_wide(
        &mut self,
        base: *mut u8,
        old_control: u8,
        new_control: u8,
        group_idx: usize,
    ) {
        if STRIDE < META_STRIDE {
            return;
        }

        let logical_len = logical_group_len(self.capacity, group_idx);
        unsafe {
            let live_ptr = base.add(META_LIVE_OFFSET);
            let tombstones_ptr = base.add(META_TOMBSTONES_OFFSET);
            let full_ptr = base.add(META_FULL_OFFSET);

            let mut live = *live_ptr;
            let mut tombstones = *tombstones_ptr;

            if old_control.is_occupied() {
                live = live.saturating_sub(1);
            } else if old_control == CTRL_TOMBSTONE {
                tombstones = tombstones.saturating_sub(1);
            }

            if new_control.is_occupied() {
                live = live.saturating_add(1);
            } else if new_control == CTRL_TOMBSTONE {
                tombstones = tombstones.saturating_add(1);
            }

            *live_ptr = live;
            *tombstones_ptr = tombstones;
            *full_ptr = u8::from(logical_len > 0 && usize::from(live) == logical_len);
        }
    }
}

/// Methods only available on wide-stride tables with embedded `GroupMeta`.
impl<T> RawTable<T, META_STRIDE> {
    /// Returns the `GroupMeta` for the given group index, reading live/tombstones/full
    /// from the metadata bytes that immediately follow the control bytes in the same block.
    #[inline]
    pub fn group_meta(&self, group_idx: usize) -> GroupMeta {
        debug_assert!(group_idx < self.group_count);
        let meta_ptr = unsafe {
            self.data_ptr
                .as_ptr()
                .add(group_idx * META_STRIDE + META_LIVE_OFFSET)
        };
        let meta_word = unsafe { meta_ptr.cast::<u32>().read_unaligned() };
        GroupMeta {
            live: (meta_word & 0xFF) as u8,
            tombstones: ((meta_word >> 8) & 0xFF) as u8,
            full: (meta_word >> 16) & 1 != 0,
        }
    }
}

#[inline]
fn offset_mask(start_offset: usize) -> u16 {
    if start_offset >= GROUP_SIZE {
        0
    } else {
        u16::MAX << start_offset
    }
}

#[inline]
fn logical_group_len(capacity: usize, group_idx: usize) -> usize {
    let group_start = group_idx * GROUP_SIZE;
    capacity.saturating_sub(group_start).min(GROUP_SIZE)
}

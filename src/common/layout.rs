use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ops::Index;
use std::ptr::{self, NonNull};

use super::control::{CTRL_EMPTY, CTRL_TOMBSTONE, ControlByte, Controls, valid_group_mask};
use super::math::round_up_to_group;

pub(crate) const GROUP_SIZE: usize = 16;
const CONTROL_ALIGNMENT: usize = 32;

#[derive(Debug)]
pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GroupMeta {
    pub(crate) live: u8,
    pub(crate) tombstones: u8,
    pub(crate) full: bool,
}

pub(crate) struct RawTable<T> {
    slots_ptr: NonNull<T>,
    controls_ptr: NonNull<u8>,
    capacity: usize,
    padded_capacity: usize,
    group_count: usize,
    groups: Box<[GroupMeta]>,
    _marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for RawTable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTable")
            .field("capacity", &self.capacity)
            .field("padded_capacity", &self.padded_capacity)
            .field("group_count", &self.group_count)
            .finish_non_exhaustive()
    }
}

impl<T> Drop for RawTable<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                alloc::dealloc(
                    self.slots_ptr.as_ptr().cast::<u8>(),
                    Self::slots_layout(self.capacity),
                );
            };
        }
        if self.padded_capacity > 0 {
            unsafe {
                alloc::dealloc(
                    self.controls_ptr.as_ptr(),
                    Self::controls_layout(self.padded_capacity),
                );
            };
        }
    }
}

impl<T> RawTable<T> {
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                slots_ptr: NonNull::dangling(),
                controls_ptr: NonNull::dangling(),
                capacity: 0,
                padded_capacity: 0,
                group_count: 0,
                groups: Box::new([]),
                _marker: PhantomData,
            };
        }

        let padded_capacity = round_up_to_group(capacity);
        let group_count = padded_capacity / GROUP_SIZE;

        let slots_layout = Self::slots_layout(capacity);
        let raw_slots = unsafe { alloc::alloc(slots_layout) };
        let slots_ptr = NonNull::new(raw_slots.cast::<T>())
            .unwrap_or_else(|| alloc::handle_alloc_error(slots_layout));

        let controls_layout = Self::controls_layout(padded_capacity);
        let raw_controls = unsafe { alloc::alloc(controls_layout) };
        let controls_ptr = NonNull::new(raw_controls)
            .unwrap_or_else(|| alloc::handle_alloc_error(controls_layout));

        unsafe { ptr::write_bytes(controls_ptr.as_ptr(), CTRL_EMPTY, padded_capacity) };

        let mut groups = vec![GroupMeta::default(); group_count].into_boxed_slice();
        for group_idx in 0..group_count {
            let logical_len = logical_group_len(capacity, group_idx);
            groups[group_idx].full = logical_len == 0;
        }

        Self {
            slots_ptr,
            controls_ptr,
            capacity,
            padded_capacity,
            group_count,
            groups,
            _marker: PhantomData,
        }
    }

    fn slots_layout(capacity: usize) -> Layout {
        Layout::array::<T>(capacity).expect("slot layout overflow")
    }

    fn controls_layout(padded_capacity: usize) -> Layout {
        Layout::from_size_align(padded_capacity, CONTROL_ALIGNMENT)
            .expect("control layout overflow")
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
    pub fn group_meta(&self, group_idx: usize) -> GroupMeta {
        self.groups[group_idx]
    }

    #[inline]
    pub fn controls<I>(&self, range: I) -> &I::Output
    where
        I: std::slice::SliceIndex<[u8]>,
    {
        self.logical_controls().index(range)
    }

    #[inline]
    pub fn logical_controls(&self) -> &[u8] {
        if self.capacity == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.controls_ptr.as_ptr(), self.capacity) }
    }

    #[inline]
    pub fn group_controls(&self, group_idx: usize) -> &[u8] {
        debug_assert!(group_idx < self.group_count);
        unsafe {
            std::slice::from_raw_parts(
                self.controls_ptr.as_ptr().add(group_idx * GROUP_SIZE),
                GROUP_SIZE,
            )
        }
    }

    #[inline]
    pub fn control_at(&self, idx: usize) -> u8 {
        unsafe { *self.controls_ptr.as_ptr().add(idx) }
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

        unsafe { *self.controls_ptr.as_ptr().add(idx) = new_control };
        self.adjust_group_meta(idx / GROUP_SIZE, old_control, new_control);
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
        if self.padded_capacity > 0 {
            unsafe {
                ptr::write_bytes(self.controls_ptr.as_ptr(), CTRL_EMPTY, self.padded_capacity);
            };
        }
        for group_idx in 0..self.group_count {
            let logical_len = self.group_len(group_idx);
            self.groups[group_idx] = GroupMeta {
                live: 0,
                tombstones: 0,
                full: logical_len == 0,
            };
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
        u16::try_from(
            self.group_controls(group_idx)
                .match_fingerprint_group(target),
        )
        .expect("group fingerprint mask fits in u16")
            & valid
    }

    #[inline]
    pub fn group_free_mask(&self, group_idx: usize) -> u16 {
        let valid = valid_group_mask(self.group_len(group_idx));
        u16::try_from(self.group_controls(group_idx).match_free_group())
            .expect("group free mask fits in u16")
            & valid
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

    fn adjust_group_meta(&mut self, group_idx: usize, old_control: u8, new_control: u8) {
        let logical_len = self.group_len(group_idx);
        let meta = &mut self.groups[group_idx];
        if old_control.is_occupied() {
            meta.live = meta.live.saturating_sub(1);
        } else if old_control == CTRL_TOMBSTONE {
            meta.tombstones = meta.tombstones.saturating_sub(1);
        }

        if new_control.is_occupied() {
            meta.live = meta.live.saturating_add(1);
        } else if new_control == CTRL_TOMBSTONE {
            meta.tombstones = meta.tombstones.saturating_add(1);
        }

        meta.full = logical_len > 0 && meta.live as usize == logical_len;
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

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vaddv_u8, vceqq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_u8, vmulq_u8, vshrq_n_u8,
};
use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ops::Index;
use std::ptr::{self, NonNull};

pub(crate) const DEFAULT_RESERVE_FRACTION: f64 = 0.10;
pub(crate) const MIN_RESERVE_FRACTION: f64 = 1e-6;
pub(crate) const MAX_RESERVE_FRACTION: f64 = 0.999_999;
pub(crate) const CTRL_EMPTY: u8 = 0;
pub(crate) const CTRL_TOMBSTONE: u8 = 0x80;

#[derive(Debug)]
pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

/// A single contiguous allocation holding both data slots and control bytes.
///
/// Layout: `[T0, T1, ..., T_{n-1}, C0, C1, ..., C_{n-1}]`
///
/// Control bytes are initialized to `CTRL_EMPTY` on creation. Slot occupancy
/// is tracked via control bytes; `RawTable` never reads a slot unless the
/// caller guarantees it has been initialized.
pub(crate) struct RawTable<T> {
    ptr: NonNull<u8>,
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for RawTable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTable")
            .field("capacity", &self.capacity)
            .finish()
    }
}

/// The `Drop` impl deallocates memory but does NOT drop contained `T` values.
/// The owning struct must drop occupied entries before dropping the `RawTable`.
impl<T> Drop for RawTable<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Self::layout(self.capacity);
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

impl<T> RawTable<T> {
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                capacity: 0,
                _marker: PhantomData,
            };
        }

        let layout = Self::layout(capacity);
        // SAFETY: layout has non-zero size because capacity > 0.
        let raw_ptr = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(raw_ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        // Initialize control bytes to CTRL_EMPTY.
        unsafe {
            let ctrl_offset = capacity * std::mem::size_of::<T>();
            let ctrl_ptr = ptr.as_ptr().add(ctrl_offset);
            ptr::write_bytes(ctrl_ptr, CTRL_EMPTY, capacity);
        }

        Self {
            ptr,
            capacity,
            _marker: PhantomData,
        }
    }

    fn layout(capacity: usize) -> Layout {
        let slots = Layout::array::<T>(capacity).expect("layout overflow");
        let ctrls = Layout::array::<u8>(capacity).expect("layout overflow");
        let (layout, _) = slots.extend(ctrls).expect("layout overflow");
        layout.pad_to_align()
    }

    #[inline]
    fn ctrl_offset(&self) -> usize {
        self.capacity * std::mem::size_of::<T>()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn controls<I>(&self, range: I) -> &I::Output
    where
        I: std::slice::SliceIndex<[u8]>,
    {
        if self.capacity == 0 {
            return [].index(range);
        }
        unsafe {
            let ctrl_ptr = self.ptr.as_ptr().add(self.ctrl_offset());
            std::slice::from_raw_parts(ctrl_ptr, self.capacity).index(range)
        }
    }

    #[inline]
    pub fn control_at(&self, idx: usize) -> u8 {
        unsafe { *self.ptr.as_ptr().add(self.ctrl_offset() + idx) }
    }

    #[inline]
    pub fn control_at_mut(&mut self, idx: usize) -> &mut u8 {
        unsafe { &mut *self.ptr.as_ptr().add(self.ctrl_offset() + idx) }
    }

    /// Write a value into the slot at `idx`, overwriting any previous content.
    #[inline]
    pub fn write(&mut self, idx: usize, value: T) {
        unsafe {
            self.ptr.as_ptr().cast::<T>().add(idx).write(value);
        }
    }

    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn get_ref(&self, idx: usize) -> &T {
        unsafe { &*self.ptr.as_ptr().cast::<T>().add(idx) }
    }

    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn get_mut(&mut self, idx: usize) -> &mut T {
        unsafe { &mut *self.ptr.as_ptr().cast::<T>().add(idx) }
    }

    /// Read the value out of the slot at `idx`, leaving it uninitialized.
    ///
    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn take(&mut self, idx: usize) -> T {
        unsafe { self.ptr.as_ptr().cast::<T>().add(idx).read() }
    }

    /// Drop the value in the slot at `idx` in place.
    ///
    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn drop_in_place(&mut self, idx: usize) {
        unsafe { ptr::drop_in_place(self.ptr.as_ptr().cast::<T>().add(idx)) }
    }
}

pub(crate) trait ControlByte {
    fn is_occupied(self) -> bool;
    fn is_free(self) -> bool;
}

impl ControlByte for u8 {
    #[inline]
    fn is_occupied(self) -> bool {
        self != CTRL_EMPTY && self != CTRL_TOMBSTONE
    }

    #[inline]
    fn is_free(self) -> bool {
        self == CTRL_EMPTY || self == CTRL_TOMBSTONE
    }
}

pub(crate) fn sanitize_reserve_fraction(reserve_fraction: f64) -> f64 {
    if reserve_fraction.is_finite() {
        reserve_fraction.clamp(MIN_RESERVE_FRACTION, MAX_RESERVE_FRACTION)
    } else {
        DEFAULT_RESERVE_FRACTION
    }
}

pub(crate) fn ceil_three_quarters(value: usize) -> usize {
    // ((3 * value) + 4 - 1) / 4
    (value.saturating_mul(3).saturating_add(3)) / 4
}

pub(crate) fn floor_half_reserve_slots(reserve_fraction: f64, value: usize) -> usize {
    ((reserve_fraction * value as f64) / 2.0).floor() as usize
}

#[inline]
pub(crate) fn control_fingerprint(hash: u64) -> u8 {
    let low = (hash as u8) & 0x7F;
    low.max(1)
}

pub(crate) fn greatest_common_divisor(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[inline]
pub(crate) fn advance_wrapping_index(current: usize, step: usize, len: usize) -> usize {
    let next = current + step;
    if next >= len { next - len } else { next }
}

pub(crate) trait Controls {
    fn find_first_free(&self) -> Option<usize>;
    fn find_first(&self, target: u8) -> Option<usize>;
    fn find_next(&self, target: u8, start: usize) -> Option<usize>;
    fn free_mask(&self) -> u16;
    fn eq_mask(&self, target: u8) -> u16;
}

impl Controls for [u8] {
    #[inline]
    fn find_first_free(&self) -> Option<usize> {
        if self.len() < 16 {
            return self.iter().position(|&control| control.is_free());
        }

        let mut index = 0usize;
        while index + 16 <= self.len() {
            let mask = self[index..index + 16].free_mask();
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += 16;
        }

        for (offset, &control) in self[index..].iter().enumerate() {
            if control.is_free() {
                return Some(index + offset);
            }
        }

        None
    }

    #[inline]
    fn find_first(&self, target: u8) -> Option<usize> {
        self.find_next(target, 0)
    }

    #[inline]
    fn find_next(&self, target: u8, start: usize) -> Option<usize> {
        if start >= self.len() {
            return None;
        }

        if self.len() - start < 16 {
            for (offset, &control) in self[start..].iter().enumerate() {
                if control == target {
                    return Some(start + offset);
                }
            }
            return None;
        }

        let mut index = start;
        while index + 16 <= self.len() {
            let mask = self[index..index + 16].eq_mask(target);
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += 16;
        }

        for (offset, &control) in self[index..].iter().enumerate() {
            if control == target {
                return Some(index + offset);
            }
        }

        None
    }

    #[inline]
    fn free_mask(&self) -> u16 {
        debug_assert!(self.len() == 16);

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { free_mask_16_neon(self.as_ptr()) }
        }

        #[cfg(target_arch = "x86_64")]
        {
            unsafe { free_mask_16_sse2(self.as_ptr()) }
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            self.eq_mask(CTRL_EMPTY) | self.eq_mask(CTRL_TOMBSTONE)
        }
    }

    #[inline]
    fn eq_mask(&self, target: u8) -> u16 {
        debug_assert!(self.len() == 16);

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { eq_mask_16_neon(self.as_ptr(), target) }
        }

        #[cfg(target_arch = "x86_64")]
        {
            unsafe { eq_mask_16_sse2(self.as_ptr(), target) }
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut mask = 0u16;
            for (idx, &value) in self.iter().enumerate() {
                if value == target {
                    mask |= 1 << idx;
                }
            }
            mask
        }
    }
}

// --- AArch64 NEON SIMD ---

/// Powers of 2 for weighted-sum bitmask extraction on NEON.
#[cfg(target_arch = "aarch64")]
static NEON_BIT_POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_movemask(cmp: core::arch::aarch64::uint8x16_t) -> u16 {
    // Convert 0xFF/0x00 lanes into a 16-bit bitmask.
    // 1. Shift right by 7: 0xFF -> 0x01, 0x00 -> 0x00
    // 2. Multiply by positional powers of 2: [1,2,4,8,...,128,1,2,4,...,128]
    // 3. Horizontal sum of low and high halves gives the bitmask bytes.
    unsafe {
        let bits = vshrq_n_u8::<7>(cmp);
        let power_vec = vld1q_u8(NEON_BIT_POWERS.as_ptr());
        let weighted = vmulq_u8(bits, power_vec);
        let lo = vaddv_u8(vget_low_u8(weighted)) as u16;
        let hi = (vaddv_u8(vget_high_u8(weighted)) as u16) << 8;
        lo | hi
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn eq_mask_16_neon(ptr: *const u8, target: u8) -> u16 {
    unsafe {
        let bytes = vld1q_u8(ptr);
        let target_vec = vdupq_n_u8(target);
        let cmp = vceqq_u8(bytes, target_vec);
        neon_movemask(cmp)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn free_mask_16_neon(ptr: *const u8) -> u16 {
    unsafe {
        let bytes = vld1q_u8(ptr);
        let empty_cmp = vceqq_u8(bytes, vdupq_n_u8(CTRL_EMPTY));
        let tombstone_cmp = vceqq_u8(bytes, vdupq_n_u8(CTRL_TOMBSTONE));
        let free_cmp = core::arch::aarch64::vorrq_u8(empty_cmp, tombstone_cmp);
        neon_movemask(free_cmp)
    }
}

// --- x86_64 SSE2 SIMD ---

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eq_mask_16_sse2(ptr: *const u8, target: u8) -> u16 {
    use std::arch::x86_64::*;
    unsafe {
        let data = _mm_loadu_si128(ptr as *const __m128i);
        let target_vec = _mm_set1_epi8(target as i8);
        let cmp = _mm_cmpeq_epi8(data, target_vec);
        _mm_movemask_epi8(cmp) as u16
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn free_mask_16_sse2(ptr: *const u8) -> u16 {
    use std::arch::x86_64::*;
    unsafe {
        let data = _mm_loadu_si128(ptr as *const __m128i);
        let empty = _mm_cmpeq_epi8(data, _mm_setzero_si128());
        let tombstone = _mm_cmpeq_epi8(data, _mm_set1_epi8(CTRL_TOMBSTONE as i8));
        let free = _mm_or_si128(empty, tombstone);
        _mm_movemask_epi8(free) as u16
    }
}

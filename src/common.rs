#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vaddv_u8, vceqq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_u8, vmulq_u8, vshrq_n_u8,
};
use std::mem::MaybeUninit;

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

/// A contiguous buffer of possibly-uninitialized entries.
///
/// Occupancy is tracked externally via control bytes; `RawSlots` never
/// reads a slot unless the caller guarantees it has been initialized.
pub(crate) struct RawSlots<T> {
    slots: Vec<MaybeUninit<T>>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for RawSlots<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawSlots")
            .field("len", &self.slots.len())
            .finish()
    }
}

impl<T> RawSlots<T> {
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        // SAFETY: MaybeUninit<T> does not require initialization.
        unsafe { slots.set_len(capacity) };
        Self { slots }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Write a value into the slot at `idx`, overwriting any previous content.
    #[inline]
    pub fn write(&mut self, idx: usize, value: T) {
        self.slots[idx] = MaybeUninit::new(value);
    }

    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn get_ref(&self, idx: usize) -> &T {
        unsafe { self.slots[idx].assume_init_ref() }
    }

    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn get_mut(&mut self, idx: usize) -> &mut T {
        unsafe { self.slots[idx].assume_init_mut() }
    }

    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn take(&mut self, idx: usize) -> T {
        unsafe { self.slots[idx].assume_init_read() }
    }

    /// # Safety
    /// The slot at `idx` must have been previously initialized via `write`.
    #[inline]
    pub unsafe fn drop_in_place(&mut self, idx: usize) {
        unsafe { self.slots[idx].assume_init_drop() }
    }
}

/// Returns true if the control byte represents an occupied slot
/// (valid fingerprint in `1..=0x7F`).
#[inline]
pub(crate) fn is_occupied_control(control: u8) -> bool {
    control != CTRL_EMPTY && control != CTRL_TOMBSTONE
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

#[inline]
pub(crate) fn is_free_control(control: u8) -> bool {
    control == CTRL_EMPTY || control == CTRL_TOMBSTONE
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

#[inline]
pub(crate) fn find_first_free_control(controls: &[u8]) -> Option<usize> {
    if controls.len() < 16 {
        return controls
            .iter()
            .position(|&control| is_free_control(control));
    }

    let mut index = 0usize;
    while index + 16 <= controls.len() {
        let mask = free_mask_16(&controls[index..index + 16]);
        if mask != 0 {
            return Some(index + mask.trailing_zeros() as usize);
        }
        index += 16;
    }

    for (offset, &control) in controls[index..].iter().enumerate() {
        if is_free_control(control) {
            return Some(index + offset);
        }
    }

    None
}

#[inline]
pub(crate) fn find_first_control(controls: &[u8], target: u8) -> Option<usize> {
    find_next_control(controls, target, 0)
}

#[inline]
pub(crate) fn find_next_control(controls: &[u8], target: u8, start: usize) -> Option<usize> {
    if start >= controls.len() {
        return None;
    }

    if controls.len() - start < 16 {
        for (offset, &control) in controls[start..].iter().enumerate() {
            if control == target {
                return Some(start + offset);
            }
        }
        return None;
    }

    let mut index = start;
    while index + 16 <= controls.len() {
        let mask = eq_mask_16(&controls[index..index + 16], target);
        if mask != 0 {
            return Some(index + mask.trailing_zeros() as usize);
        }
        index += 16;
    }

    for (offset, &control) in controls[index..].iter().enumerate() {
        if control == target {
            return Some(index + offset);
        }
    }

    None
}

#[inline]
fn free_mask_16(chunk: &[u8]) -> u16 {
    debug_assert!(chunk.len() == 16);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `chunk` has exactly 16 bytes by construction.
        unsafe { free_mask_16_neon(chunk.as_ptr()) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe { free_mask_16_sse2(chunk.as_ptr()) }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        eq_mask_16_scalar(chunk, CTRL_EMPTY) | eq_mask_16_scalar(chunk, CTRL_TOMBSTONE)
    }
}

#[inline]
fn eq_mask_16(chunk: &[u8], target: u8) -> u16 {
    debug_assert!(chunk.len() == 16);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `chunk` has exactly 16 bytes by construction.
        unsafe { eq_mask_16_neon(chunk.as_ptr(), target) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe { eq_mask_16_sse2(chunk.as_ptr(), target) }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        eq_mask_16_scalar(chunk, target)
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn eq_mask_16_scalar(chunk: &[u8], target: u8) -> u16 {
    let mut mask = 0u16;
    for (idx, &value) in chunk.iter().enumerate() {
        if value == target {
            mask |= 1 << idx;
        }
    }
    mask
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

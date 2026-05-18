#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    uint8x16_t, vandq_u8, vceqq_u8, vdupq_n_u8, vget_lane_u64, vld1q_u8, vreinterpret_u64_u8,
    vreinterpretq_u16_u8, vshrn_n_u16,
};
#[cfg(target_arch = "x86_64")]
use {
    core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch},
    std::arch::x86_64::{
        __m128i, __m256i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8,
        _mm_set1_epi8, _mm_setzero_si128, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_set1_epi8,
    },
    std::sync::OnceLock,
};

use super::bitmask::BitMask;

pub(crate) const CONTROL_GROUP_SIZE: usize = 16;
pub(crate) const CTRL_EMPTY: u8 = 0;
pub(crate) const CTRL_TOMBSTONE: u8 = 0x80;
/// Low 7 bits hold the fingerprint; high bit distinguishes occupied (0) from
/// the tombstone sentinel (`CTRL_TOMBSTONE`).
pub(crate) const FINGERPRINT_MASK: u8 = 0x7F;
/// Shift that pulls the 7 high bits of a 64-bit hash into bits [6:0].
const FINGERPRINT_SHIFT: u32 = 57;

pub(crate) trait ControlByte {
    fn is_occupied(&self) -> bool;
    fn is_free(&self) -> bool;
}

impl ControlByte for u8 {
    #[inline]
    fn is_occupied(&self) -> bool {
        (*self & FINGERPRINT_MASK) != 0
    }

    #[inline]
    fn is_free(&self) -> bool {
        (*self & FINGERPRINT_MASK) == 0
    }
}

// ---------------------------------------------------------------------------
// ControlOps — namespace for control-byte static methods
// ---------------------------------------------------------------------------

pub(crate) struct ControlOps;

impl ControlOps {
    #[inline]
    #[must_use]
    pub(crate) fn control_fingerprint(hash: u64) -> u8 {
        // Masking with FINGERPRINT_MASK (0x7F) bounds the value to [0, 127],
        // so the truncating `as u8` cast is lossless.
        #[allow(clippy::cast_possible_truncation)]
        let high = ((hash >> FINGERPRINT_SHIFT) & u64::from(FINGERPRINT_MASK)) as u8;
        high.max(1)
    }

    #[inline]
    #[must_use]
    pub(crate) fn fingerprint_bit(fingerprint: u8) -> u128 {
        1u128 << u32::from(fingerprint.saturating_sub(1))
    }

    #[inline]
    #[must_use]
    pub(crate) fn find_next_fingerprint_in_controls(
        controls: &[u8],
        fingerprint: u8,
        start: usize,
    ) -> Option<usize> {
        if start >= controls.len() {
            return None;
        }

        if controls.len() - start < CONTROL_GROUP_SIZE {
            return controls[start..]
                .iter()
                .position(|&control| control == fingerprint)
                .map(|offset| start + offset);
        }

        let wide = Self::preferred_group_width();
        let mut index = start;
        while wide > CONTROL_GROUP_SIZE && index + wide <= controls.len() {
            let mask =
                Self::control_match_fingerprint_group(&controls[index..index + wide], fingerprint);
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += wide;
        }

        while index + CONTROL_GROUP_SIZE <= controls.len() {
            let mask = Self::control_match_fingerprint_group(
                &controls[index..index + CONTROL_GROUP_SIZE],
                fingerprint,
            );
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += CONTROL_GROUP_SIZE;
        }

        controls[index..]
            .iter()
            .position(|&control| control == fingerprint)
            .map(|offset| index + offset)
    }

    #[inline]
    #[must_use]
    fn preferred_group_width() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            static WIDTH: OnceLock<usize> = OnceLock::new();
            *WIDTH.get_or_init(|| {
                if std::is_x86_feature_detected!("avx2") {
                    32
                } else {
                    CONTROL_GROUP_SIZE
                }
            })
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            CONTROL_GROUP_SIZE
        }
    }

    /// Returns a 1-bit-per-byte u32 mask for `find_next_fingerprint_in_controls`.
    /// This is a cold-path fallback; performance-critical callers use `eq_mask_16`
    /// which returns the arch-native `BitMask`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk` is not 16 or 32 bytes long.
    #[inline]
    #[must_use]
    pub(crate) fn control_match_fingerprint_group(chunk: &[u8], target: u8) -> u32 {
        match chunk.len() {
            CONTROL_GROUP_SIZE => match_fingerprint_group_u32(chunk.as_ptr(), target),
            32 => unsafe { eq_mask_32(chunk.as_ptr(), target) },
            _ => panic!("group matching requires 16 or 32 byte chunks"),
        }
    }
}

/// 1-bit-per-byte u32 mask over a 16-byte chunk. Cold fallback path only.
#[inline]
fn match_fingerprint_group_u32(ptr: *const u8, target: u8) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let data = _mm_loadu_si128(ptr.cast::<__m128i>());
        #[allow(clippy::cast_possible_wrap)]
        let cmp = _mm_cmpeq_epi8(data, _mm_set1_epi8(target as i8));
        #[allow(clippy::cast_sign_loss)]
        {
            (_mm_movemask_epi8(cmp) as u32) & 0xFFFF
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut m = 0u32;
        for i in 0..CONTROL_GROUP_SIZE {
            if unsafe { *ptr.add(i) } == target {
                m |= 1 << i;
            }
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Probe helpers (ProbeOps)
// ---------------------------------------------------------------------------

pub(crate) struct ProbeOps;

impl ProbeOps {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[inline]
    #[must_use]
    pub(crate) fn log_log_probe_limit(capacity: usize) -> usize {
        let n = capacity.max(4) as f64;
        n.log2().max(2.0).log2().ceil().max(1.0) as usize
    }

    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub(crate) fn hash_to_usize(hash: u64) -> usize {
        hash as usize
    }
}

// ---------------------------------------------------------------------------
// SIMD mask functions
// ---------------------------------------------------------------------------

/// # Safety
///
/// `ptr` must be valid to read `CONTROL_GROUP_SIZE` bytes.
#[must_use]
pub(crate) unsafe fn eq_mask_16(ptr: *const u8, target: u8) -> BitMask {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        eq_mask_16_neon(ptr, target)
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        eq_mask_16_sse2(ptr, target)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut m: u16 = 0;
        for i in 0..CONTROL_GROUP_SIZE {
            if unsafe { *ptr.add(i) } == target {
                m |= 1u16 << i;
            }
        }
        BitMask(m)
    }
}

/// # Safety
///
/// `ptr` must be valid to read `CONTROL_GROUP_SIZE` bytes.
#[must_use]
pub(crate) unsafe fn free_mask_16(ptr: *const u8) -> BitMask {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        free_mask_16_neon(ptr)
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        free_mask_16_sse2(ptr)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut m: u16 = 0;
        for i in 0..CONTROL_GROUP_SIZE {
            let b = unsafe { *ptr.add(i) };
            if b == CTRL_EMPTY || b == CTRL_TOMBSTONE {
                m |= 1u16 << i;
            }
        }
        BitMask(m)
    }
}

/// # Safety
///
/// `ptr` must be valid to read 32 bytes.
#[must_use]
pub(crate) unsafe fn eq_mask_32(ptr: *const u8, target: u8) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return eq_mask_32_avx2(ptr, target) };
        }
    }

    let lo = match_fingerprint_group_u32(ptr, target);
    let hi = match_fingerprint_group_u32(unsafe { ptr.add(CONTROL_GROUP_SIZE) }, target);
    lo | (hi << CONTROL_GROUP_SIZE)
}

// ---------------------------------------------------------------------------
// Prefetch
// ---------------------------------------------------------------------------

/// # Safety
///
/// `ptr` must be a valid, aligned pointer to readable memory (or null, in which
/// case the prefetch is silently ignored by the hardware).
#[inline]
pub(crate) unsafe fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{}]", in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }
}

// ---------------------------------------------------------------------------
// Platform-specific SIMD implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn nibble_mask_from_cmp(cmp: uint8x16_t) -> BitMask {
    // vshrn narrows 16×u8 → 8 bytes: each source byte of 0xFF becomes a nibble
    // of 0xF in the output, 0x00 becomes 0x0. Result u64 has 4 bits per slot.
    unsafe {
        let narrowed = vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4);
        BitMask(vget_lane_u64(vreinterpret_u64_u8(narrowed), 0))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn eq_mask_16_neon(ptr: *const u8, target: u8) -> BitMask {
    unsafe {
        let bytes = vld1q_u8(ptr);
        let cmp = vceqq_u8(bytes, vdupq_n_u8(target));
        nibble_mask_from_cmp(cmp)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn free_mask_16_neon(ptr: *const u8) -> BitMask {
    unsafe {
        let bytes = vld1q_u8(ptr);
        let masked = vandq_u8(bytes, vdupq_n_u8(FINGERPRINT_MASK));
        let free_cmp = vceqq_u8(masked, vdupq_n_u8(0));
        nibble_mask_from_cmp(free_cmp)
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eq_mask_16_sse2(ptr: *const u8, target: u8) -> BitMask {
    unsafe {
        let data = _mm_loadu_si128(ptr.cast::<__m128i>());
        let target_vec = _mm_set1_epi8(target as i8);
        let cmp = _mm_cmpeq_epi8(data, target_vec);
        #[allow(clippy::cast_possible_truncation)]
        {
            BitMask(_mm_movemask_epi8(cmp) as u16)
        }
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn free_mask_16_sse2(ptr: *const u8) -> BitMask {
    unsafe {
        let data = _mm_loadu_si128(ptr.cast::<__m128i>());
        let masked = _mm_and_si128(data, _mm_set1_epi8(FINGERPRINT_MASK as i8));
        let free = _mm_cmpeq_epi8(masked, _mm_setzero_si128());
        #[allow(clippy::cast_possible_truncation)]
        {
            BitMask(_mm_movemask_epi8(free) as u16)
        }
    }
}

#[allow(clippy::cast_possible_wrap)]
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eq_mask_32_avx2(ptr: *const u8, target: u8) -> u32 {
    unsafe {
        let data = _mm256_loadu_si256(ptr.cast::<__m256i>());
        let target_vec = _mm256_set1_epi8(target as i8);
        let cmp = _mm256_cmpeq_epi8(data, target_vec);
        #[allow(clippy::cast_sign_loss)]
        {
            _mm256_movemask_epi8(cmp) as u32
        }
    }
}

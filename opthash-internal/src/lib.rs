#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vaddv_u8, vandq_u8, vceqq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_u8,
};
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

pub const CONTROL_GROUP_SIZE: usize = 16;
pub const CTRL_EMPTY: u8 = 0;
pub const CTRL_TOMBSTONE: u8 = 0x80;

pub trait ControlByte {
    fn is_occupied(&self) -> bool;
    fn is_free(&self) -> bool;
}

impl ControlByte for u8 {
    #[inline]
    fn is_occupied(&self) -> bool {
        *self != CTRL_EMPTY && *self != CTRL_TOMBSTONE
    }

    #[inline]
    fn is_free(&self) -> bool {
        *self == CTRL_EMPTY || *self == CTRL_TOMBSTONE
    }
}

// ---------------------------------------------------------------------------
// ControlOps — namespace for control-byte static methods
// ---------------------------------------------------------------------------

pub struct ControlOps;

impl ControlOps {
    /// # Panics
    ///
    /// Panics if the masked 7-bit fingerprint cannot be represented as `u8`.
    #[inline]
    #[must_use]
    pub fn control_fingerprint(hash: u64) -> u8 {
        let high = u8::try_from((hash >> 57) & 0x7F).expect("7-bit fingerprint fits in u8");
        high.max(1)
    }

    #[inline]
    #[must_use]
    pub fn fingerprint_bit(fingerprint: u8) -> u128 {
        1u128 << u32::from(fingerprint.saturating_sub(1))
    }

    #[inline]
    #[must_use]
    pub fn find_first_free_in_controls(controls: &[u8]) -> Option<usize> {
        if controls.len() < CONTROL_GROUP_SIZE {
            return controls
                .iter()
                .position(|&c| c == CTRL_EMPTY || c == CTRL_TOMBSTONE);
        }

        let wide = Self::preferred_group_width();
        let mut index = 0;
        while wide > CONTROL_GROUP_SIZE && index + wide <= controls.len() {
            let mask = Self::control_match_free_group(&controls[index..index + wide]);
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += wide;
        }

        while index + CONTROL_GROUP_SIZE <= controls.len() {
            let mask = Self::control_match_free_group(&controls[index..index + CONTROL_GROUP_SIZE]);
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += CONTROL_GROUP_SIZE;
        }

        controls[index..]
            .iter()
            .position(|&c| c == CTRL_EMPTY || c == CTRL_TOMBSTONE)
            .map(|offset| index + offset)
    }

    #[inline]
    #[must_use]
    pub fn find_next_fingerprint_in_controls(
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

    /// # Panics
    ///
    /// Panics if `chunk` is not 16 or 32 bytes long.
    #[inline]
    #[must_use]
    pub fn control_match_fingerprint_group(chunk: &[u8], target: u8) -> u32 {
        match chunk.len() {
            CONTROL_GROUP_SIZE => u32::from(unsafe { eq_mask_16(chunk.as_ptr(), target) }),
            32 => unsafe { eq_mask_32(chunk.as_ptr(), target) },
            _ => panic!("group matching requires 16 or 32 byte chunks"),
        }
    }

    /// # Panics
    ///
    /// Panics if `chunk` is not 16 or 32 bytes long.
    #[inline]
    #[must_use]
    pub fn control_match_free_group(chunk: &[u8]) -> u32 {
        match chunk.len() {
            CONTROL_GROUP_SIZE => u32::from(unsafe { free_mask_16(chunk.as_ptr()) }),
            32 => unsafe { free_mask_32(chunk.as_ptr()) },
            _ => panic!("group matching requires 16 or 32 byte chunks"),
        }
    }
}

// ---------------------------------------------------------------------------
// Probe helpers (ProbeOps)
// ---------------------------------------------------------------------------

pub struct ProbeOps;

impl ProbeOps {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[inline]
    #[must_use]
    pub fn log_log_probe_limit(capacity: usize) -> usize {
        let n = capacity.max(4) as f64;
        n.log2().max(2.0).log2().ceil().max(1.0) as usize
    }

    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    #[must_use]
    pub fn hash_to_usize(hash: u64) -> usize {
        hash as usize
    }

    #[inline]
    #[must_use]
    pub fn greatest_common_divisor(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let next = a % b;
            a = b;
            b = next;
        }
        a
    }

    #[inline]
    #[must_use]
    pub fn advance_wrapping_index(index: usize, step: usize, len: usize) -> usize {
        // step < len is guaranteed by build_group_steps, so index + step < 2*len.
        // A conditional subtract avoids the expensive division that modulo requires.
        if len == 0 {
            return 0;
        }
        let r = index + step;
        if r >= len { r - len } else { r }
    }

    #[must_use]
    pub fn build_group_steps(group_count: usize) -> Box<[usize]> {
        if group_count <= 1 {
            return Box::new([1]);
        }

        let mut steps = Vec::new();
        for step in 1..group_count {
            if Self::greatest_common_divisor(step, group_count) == 1 {
                steps.push(step);
            }
        }
        if steps.is_empty() {
            steps.push(1);
        }
        steps.into_boxed_slice()
    }
}

// ---------------------------------------------------------------------------
// SIMD mask functions
// ---------------------------------------------------------------------------

/// # Safety
///
/// `ptr` must be valid to read `CONTROL_GROUP_SIZE` bytes.
#[must_use]
#[cfg(target_arch = "aarch64")]
pub unsafe fn eq_mask_16(ptr: *const u8, target: u8) -> u16 {
    unsafe { eq_mask_16_neon(ptr, target) }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn eq_mask_16(ptr: *const u8, target: u8) -> u16 {
    unsafe { eq_mask_16_sse2(ptr, target) }
}

/// # Safety
///
/// `ptr` must be valid to read `CONTROL_GROUP_SIZE` bytes.
#[must_use]
#[cfg(target_arch = "aarch64")]
pub unsafe fn free_mask_16(ptr: *const u8) -> u16 {
    unsafe { free_mask_16_neon(ptr) }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn free_mask_16(ptr: *const u8) -> u16 {
    unsafe { free_mask_16_sse2(ptr) }
}

/// # Safety
///
/// `ptr` must be valid to read 32 bytes.
#[must_use]
pub unsafe fn eq_mask_32(ptr: *const u8, target: u8) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return eq_mask_32_avx2(ptr, target) };
        }
    }

    let lo = u32::from(unsafe { eq_mask_16(ptr, target) });
    let hi = u32::from(unsafe { eq_mask_16(ptr.add(CONTROL_GROUP_SIZE), target) });
    lo | (hi << CONTROL_GROUP_SIZE)
}

/// # Safety
///
/// `ptr` must be valid to read 32 bytes.
#[must_use]
pub unsafe fn free_mask_32(ptr: *const u8) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return free_mask_32_avx2(ptr) };
        }
    }

    let lo = u32::from(unsafe { free_mask_16(ptr) });
    let hi = u32::from(unsafe { free_mask_16(ptr.add(CONTROL_GROUP_SIZE)) });
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
pub unsafe fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{}]", in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
}

// ---------------------------------------------------------------------------
// Platform-specific SIMD implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
static NEON_BIT_POWERS: [u8; CONTROL_GROUP_SIZE] =
    [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_movemask(cmp: core::arch::aarch64::uint8x16_t) -> u16 {
    unsafe {
        let power_vec = vld1q_u8(NEON_BIT_POWERS.as_ptr());
        let weighted = vandq_u8(cmp, power_vec);
        let lo = u16::from(vaddv_u8(vget_low_u8(weighted)));
        let hi = u16::from(vaddv_u8(vget_high_u8(weighted))) << 8;
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

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eq_mask_16_sse2(ptr: *const u8, target: u8) -> u16 {
    use std::arch::x86_64::*;
    unsafe {
        let data = _mm_loadu_si128(ptr.cast::<__m128i>());
        let target_vec = _mm_set1_epi8(target as i8);
        let cmp = _mm_cmpeq_epi8(data, target_vec);
        #[allow(clippy::cast_possible_truncation)]
        {
            _mm_movemask_epi8(cmp) as u16
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn free_mask_16_sse2(ptr: *const u8) -> u16 {
    use std::arch::x86_64::*;
    unsafe {
        let data = _mm_loadu_si128(ptr.cast::<__m128i>());
        let empty = _mm_cmpeq_epi8(data, _mm_setzero_si128());
        let tombstone = _mm_cmpeq_epi8(data, _mm_set1_epi8(CTRL_TOMBSTONE as i8));
        let free = _mm_or_si128(empty, tombstone);
        #[allow(clippy::cast_possible_truncation)]
        {
            _mm_movemask_epi8(free) as u16
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eq_mask_32_avx2(ptr: *const u8, target: u8) -> u32 {
    use std::arch::x86_64::*;
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

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn free_mask_32_avx2(ptr: *const u8) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let data = _mm256_loadu_si256(ptr.cast::<__m256i>());
        let empty = _mm256_cmpeq_epi8(data, _mm256_setzero_si256());
        let tombstone = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(CTRL_TOMBSTONE as i8));
        let free = _mm256_or_si256(empty, tombstone);
        #[allow(clippy::cast_sign_loss)]
        {
            _mm256_movemask_epi8(free) as u32
        }
    }
}

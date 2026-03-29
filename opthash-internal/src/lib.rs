#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vaddv_u8, vceqq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_u8, vmulq_u8, vshrq_n_u8,
};
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

pub const CONTROL_GROUP_SIZE: usize = 16;
pub const CTRL_EMPTY: u8 = 0;
pub const CTRL_TOMBSTONE: u8 = 0x80;

#[must_use]
pub fn preferred_group_width() -> usize {
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
#[must_use]
pub fn control_match_free_group(chunk: &[u8]) -> u32 {
    match chunk.len() {
        CONTROL_GROUP_SIZE => u32::from(unsafe { free_mask_16(chunk.as_ptr()) }),
        32 => unsafe { free_mask_32(chunk.as_ptr()) },
        _ => panic!("group matching requires 16 or 32 byte chunks"),
    }
}

/// # Panics
///
/// Panics if `chunk` is not 16 or 32 bytes long.
#[must_use]
pub fn control_match_fingerprint_group(chunk: &[u8], target: u8) -> u32 {
    match chunk.len() {
        CONTROL_GROUP_SIZE => u32::from(unsafe { eq_mask_16(chunk.as_ptr(), target) }),
        32 => unsafe { eq_mask_32(chunk.as_ptr(), target) },
        _ => panic!("group matching requires 16 or 32 byte chunks"),
    }
}

/// # Safety
///
/// `ptr` must be valid to read `CONTROL_GROUP_SIZE` bytes.
#[must_use]
pub unsafe fn eq_mask_16(ptr: *const u8, target: u8) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { return eq_mask_16_neon(ptr, target) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe { return eq_mask_16_sse2(ptr, target) };
    }

    #[allow(unreachable_code)]
    {
        let slice = unsafe { std::slice::from_raw_parts(ptr, CONTROL_GROUP_SIZE) };
        let mut mask = 0u16;
        for (idx, &value) in slice.iter().enumerate() {
            if value == target {
                mask |= 1 << idx;
            }
        }
        mask
    }
}

/// # Safety
///
/// `ptr` must be valid to read `CONTROL_GROUP_SIZE` bytes.
#[must_use]
pub unsafe fn free_mask_16(ptr: *const u8) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { return free_mask_16_neon(ptr) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe { return free_mask_16_sse2(ptr) };
    }

    #[allow(unreachable_code)]
    {
        unsafe { eq_mask_16(ptr, CTRL_EMPTY) | eq_mask_16(ptr, CTRL_TOMBSTONE) }
    }
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

#[cfg(target_arch = "aarch64")]
static NEON_BIT_POWERS: [u8; CONTROL_GROUP_SIZE] =
    [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_movemask(cmp: core::arch::aarch64::uint8x16_t) -> u16 {
    unsafe {
        let bits = vshrq_n_u8::<7>(cmp);
        let power_vec = vld1q_u8(NEON_BIT_POWERS.as_ptr());
        let weighted = vmulq_u8(bits, power_vec);
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

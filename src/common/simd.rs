#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vaddv_u8, vceqq_u8, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_u8, vmulq_u8, vshrq_n_u8,
};
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

use super::control::{CTRL_EMPTY, CTRL_TOMBSTONE};
use super::layout::GROUP_SIZE;

pub(super) fn preferred_group_width() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        static WIDTH: OnceLock<usize> = OnceLock::new();
        *WIDTH.get_or_init(|| {
            if std::is_x86_feature_detected!("avx2") {
                32
            } else {
                GROUP_SIZE
            }
        })
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        GROUP_SIZE
    }
}

pub(super) fn eq_mask_16(ptr: *const u8, target: u8) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { eq_mask_16_neon(ptr, target) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe { eq_mask_16_sse2(ptr, target) }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let slice = unsafe { std::slice::from_raw_parts(ptr, GROUP_SIZE) };
        let mut mask = 0u16;
        for (idx, &value) in slice.iter().enumerate() {
            if value == target {
                mask |= 1 << idx;
            }
        }
        mask
    }
}

pub(super) fn free_mask_16(ptr: *const u8) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { free_mask_16_neon(ptr) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        unsafe { free_mask_16_sse2(ptr) }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        eq_mask_16(ptr, CTRL_EMPTY) | eq_mask_16(ptr, CTRL_TOMBSTONE)
    }
}

pub(super) fn eq_mask_32(ptr: *const u8, target: u8) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return eq_mask_32_avx2(ptr, target) };
        }
    }

    let lo = eq_mask_16(ptr, target) as u32;
    let hi = eq_mask_16(unsafe { ptr.add(GROUP_SIZE) }, target) as u32;
    lo | (hi << GROUP_SIZE)
}

pub(super) fn free_mask_32(ptr: *const u8) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return free_mask_32_avx2(ptr) };
        }
    }

    let lo = free_mask_16(ptr) as u32;
    let hi = free_mask_16(unsafe { ptr.add(GROUP_SIZE) }) as u32;
    lo | (hi << GROUP_SIZE)
}

#[cfg(target_arch = "aarch64")]
static NEON_BIT_POWERS: [u8; GROUP_SIZE] =
    [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_movemask(cmp: core::arch::aarch64::uint8x16_t) -> u16 {
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

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eq_mask_32_avx2(ptr: *const u8, target: u8) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let data = _mm256_loadu_si256(ptr as *const __m256i);
        let target_vec = _mm256_set1_epi8(target as i8);
        let cmp = _mm256_cmpeq_epi8(data, target_vec);
        _mm256_movemask_epi8(cmp) as u32
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn free_mask_32_avx2(ptr: *const u8) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let data = _mm256_loadu_si256(ptr as *const __m256i);
        let empty = _mm256_cmpeq_epi8(data, _mm256_setzero_si256());
        let tombstone = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(CTRL_TOMBSTONE as i8));
        let free = _mm256_or_si256(empty, tombstone);
        _mm256_movemask_epi8(free) as u32
    }
}

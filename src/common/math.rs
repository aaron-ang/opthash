use super::config::{DEFAULT_RESERVE_FRACTION, MAX_RESERVE_FRACTION, MIN_RESERVE_FRACTION};
use super::layout::GROUP_SIZE;

pub(crate) use super::simd::ProbeOps;

#[allow(clippy::cast_precision_loss)]
#[inline]
pub(crate) fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[inline]
pub(crate) fn floor_to_usize(value: f64) -> usize {
    value.floor() as usize
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[inline]
pub(crate) fn ceil_to_usize(value: f64) -> usize {
    value.ceil() as usize
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[inline]
pub(crate) fn round_to_usize(value: f64) -> usize {
    value.round() as usize
}

#[inline]
pub(crate) fn max_insertions(capacity: usize, reserve_fraction: f64) -> usize {
    capacity.saturating_sub(floor_to_usize(reserve_fraction * usize_to_f64(capacity)))
}

pub(crate) fn sanitize_reserve_fraction(reserve_fraction: f64) -> f64 {
    if reserve_fraction.is_finite() {
        reserve_fraction.clamp(MIN_RESERVE_FRACTION, MAX_RESERVE_FRACTION)
    } else {
        DEFAULT_RESERVE_FRACTION
    }
}

pub(crate) fn ceil_three_quarters(value: usize) -> usize {
    (value.saturating_mul(3).saturating_add(3)) / 4
}

pub(crate) fn floor_half_reserve_slots(reserve_fraction: f64, value: usize) -> usize {
    floor_to_usize((reserve_fraction * usize_to_f64(value)) / 2.0)
}

#[inline]
pub(crate) fn advance_wrapping_index(current: usize, step: usize, len: usize) -> usize {
    ProbeOps::advance_wrapping_index(current, step, len)
}

#[inline]
pub(crate) fn round_up_to_group(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        value.div_ceil(GROUP_SIZE) * GROUP_SIZE
    }
}

/// Knuth multiplicative-hash constant (golden ratio * 2^64, odd).
const GOLDEN_RATIO_U64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Per-level hash salt: mixes the level index into the hash to decorrelate
/// bucket distributions across levels. Used by both elastic and funnel.
#[inline]
pub(crate) fn level_salt(level_idx: usize) -> u64 {
    GOLDEN_RATIO_U64.wrapping_mul(
        u64::try_from(level_idx)
            .expect("level index fits in u64")
            .wrapping_add(1),
    )
}

/// Lemire fastmod: precompute magic `M = ceil(2^64 / d)` once so that each
/// later `a % d` becomes a multiply-shift. `d` must be non-zero and fit in
/// `u32` (panics otherwise).
#[inline]
#[must_use]
pub(crate) fn fastmod_magic(d: usize) -> u64 {
    let d_u32 = u32::try_from(d).expect("fastmod divisor fits in u32");
    debug_assert!(d_u32 != 0);
    (u64::MAX / u64::from(d_u32)).wrapping_add(1)
}

/// Evaluates `(a mod d)` using the magic produced by [`fastmod_magic`]. `a`
/// contributes through its low 32 bits; `d` is the same value originally
/// passed to `fastmod_magic`. All narrowing casts are safe given that
/// precondition (see inline comments).
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn fastmod_u32(a: u64, m: u64, d: usize) -> usize {
    // `d` fits in u32 by the magic-builder precondition, so the truncation is
    // exact. Only the low 32 bits of `a` participate in Lemire's trick; we
    // sample them explicitly so callers don't need to narrow first.
    let d_u32 = d as u32;
    let a_lo = a as u32;
    let lowbits = m.wrapping_mul(u64::from(a_lo));
    // lowbits < 2^64 and d < 2^32, so (lowbits * d) < 2^96 and the final
    // shift-right-by-64 yields a value that fits in u32.
    let result = ((u128::from(lowbits) * u128::from(d_u32)) >> 64) as u32;
    result as usize
}

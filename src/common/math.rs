use super::config::{DEFAULT_RESERVE_FRACTION, MAX_RESERVE_FRACTION, MIN_RESERVE_FRACTION};
use super::layout::GROUP_SIZE;

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
    ((reserve_fraction * value as f64) / 2.0).floor() as usize
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
pub(crate) fn round_up_to_group(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        value.div_ceil(GROUP_SIZE) * GROUP_SIZE
    }
}

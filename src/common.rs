pub(crate) const DEFAULT_CAPACITY: usize = 1024;
pub(crate) const DEFAULT_RESERVE_FRACTION: f64 = 0.10;
pub(crate) const MIN_RESERVE_FRACTION: f64 = 1e-6;
pub(crate) const MAX_RESERVE_FRACTION: f64 = 0.999_999;

#[derive(Debug)]
pub(crate) struct Entry<K, V> {
    pub(crate) key: K,
    pub(crate) value: V,
}

pub(crate) fn empty_slots<T>(capacity: usize) -> Vec<Option<T>> {
    let mut slots = Vec::with_capacity(capacity);
    slots.resize_with(capacity, || None);
    slots
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

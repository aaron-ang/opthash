pub(crate) mod bitmask;
pub(crate) mod config;
pub(crate) mod control;
pub(crate) mod layout;
pub(crate) mod math;
pub(crate) mod simd;

pub type DefaultHashBuilder = foldhash::fast::RandomState;

/// Error returned by `try_reserve` when the map can't grow.
///
/// Mirrors the role of [`std::collections::TryReserveError`]. We use a local
/// type because std's error has private fields and no stable constructor,
/// so library code can't build one from a raw allocation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TryReserveError {
    /// Computing the new capacity overflowed `usize`.
    CapacityOverflow,
    /// The allocator failed to satisfy the request.
    AllocError,
}

impl std::fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapacityOverflow => f.write_str("capacity overflow"),
            Self::AllocError => f.write_str("memory allocation failed"),
        }
    }
}

impl std::error::Error for TryReserveError {}

#[cfg(test)]
mod tests {
    use super::layout::RawTable;

    #[test]
    fn group_masks_work_on_full_groups() {
        let mut table: RawTable<u64> = RawTable::new(32);
        table.set_control(16, 11);
        assert_eq!(table.group_match_mask(1, 11).lowest(), Some(0));
        assert!(table.group_free_mask(1).any());
    }
}

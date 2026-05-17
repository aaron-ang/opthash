pub(crate) mod bitmask;
pub(crate) mod config;
pub(crate) mod control;
pub(crate) mod layout;
pub(crate) mod math;
pub(crate) mod simd;

pub type DefaultHashBuilder = foldhash::fast::RandomState;

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

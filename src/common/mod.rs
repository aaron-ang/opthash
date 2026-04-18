pub(crate) mod bitmask;
pub(crate) mod config;
pub(crate) mod control;
pub(crate) mod layout;
pub(crate) mod math;
pub(crate) mod simd;

#[cfg(test)]
mod tests {
    use super::control::ControlOps;
    use super::layout::RawTable;

    #[test]
    fn group_masks_work_on_full_groups() {
        let mut table: RawTable<u64> = RawTable::new(32);
        table.set_control(16, 11);
        assert_eq!(table.group_match_mask(1, 11).lowest(), Some(0));
        assert!(table.group_free_mask(1).any());
    }

    #[test]
    fn fingerprint_summary_bit_matches_fingerprint_domain() {
        let fp = ControlOps::control_fingerprint(0xDEAD_BEEF);
        let bit = ControlOps::fingerprint_bit(fp);
        assert_ne!(bit, 0);
    }
}

pub(crate) mod config;
pub(crate) mod control;
pub(crate) mod layout;
pub(crate) mod math;

#[cfg(test)]
mod tests {
    use super::control::{CTRL_EMPTY, Controls, control_fingerprint, fingerprint_bit};
    use super::layout::RawTable;

    #[test]
    fn control_group_matches_fingerprints() {
        let mut controls = [7u8; 32];
        controls[0] = 9;
        controls[17] = 9;
        controls[18] = 3;

        let expected = (1u32 << 0) | (1u32 << 17);
        assert_eq!(controls[..32].match_fingerprint_group(9), expected);
    }

    #[test]
    fn padded_controls_are_empty_sentinels() {
        let table: RawTable<u64> = RawTable::new(18);
        let last_group = table.group_controls(1);
        assert_eq!(last_group[2..], [CTRL_EMPTY; 14]);
    }

    #[test]
    fn group_masks_work_on_full_groups() {
        let mut table: RawTable<u64> = RawTable::new(32);
        table.set_control(16, 11);
        assert_eq!(table.group_match_mask(1, 11), 0b1);
        assert!(table.group_free_mask(1) != 0);
    }

    #[test]
    fn fingerprint_summary_bit_matches_fingerprint_domain() {
        let fp = control_fingerprint(0xDEAD_BEEF);
        let bit = fingerprint_bit(fp);
        assert_ne!(bit, 0);
    }
}

use super::layout::GROUP_SIZE;
use opthash_internal::{ControlOps, eq_mask_16, eq_mask_32, free_mask_16, free_mask_32};

pub(crate) use opthash_internal::{CTRL_EMPTY, CTRL_TOMBSTONE};

pub(crate) trait ControlByte {
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

#[inline]
pub(crate) fn control_fingerprint(hash: u64) -> u8 {
    ControlOps::control_fingerprint(hash)
}

#[inline]
pub(crate) fn fingerprint_bit(fingerprint: u8) -> u128 {
    ControlOps::fingerprint_bit(fingerprint)
}

#[inline]
pub(crate) fn valid_group_mask(len: usize) -> u16 {
    if len >= GROUP_SIZE {
        u16::MAX
    } else if len == 0 {
        0
    } else {
        (1u16 << len) - 1
    }
}

pub(crate) trait Controls {
    fn find_first_free(&self) -> Option<usize>;
    fn find_first(&self, target: u8) -> Option<usize>;
    fn find_next(&self, target: u8, start: usize) -> Option<usize>;
    fn match_free_group(&self) -> u32;
    fn match_fingerprint_group(&self, target: u8) -> u32;
}

impl Controls for [u8] {
    #[inline]
    fn find_first_free(&self) -> Option<usize> {
        ControlOps::find_first_free_in_controls(self)
    }

    #[inline]
    fn find_first(&self, target: u8) -> Option<usize> {
        self.find_next(target, 0)
    }

    #[inline]
    fn find_next(&self, target: u8, start: usize) -> Option<usize> {
        ControlOps::find_next_fingerprint_in_controls(self, target, start)
    }

    #[inline]
    fn match_free_group(&self) -> u32 {
        match self.len() {
            GROUP_SIZE => u32::from(unsafe { free_mask_16(self.as_ptr()) }),
            32 => unsafe { free_mask_32(self.as_ptr()) },
            _ => panic!("group matching requires 16 or 32 byte chunks"),
        }
    }

    #[inline]
    fn match_fingerprint_group(&self, target: u8) -> u32 {
        match self.len() {
            GROUP_SIZE => u32::from(unsafe { eq_mask_16(self.as_ptr(), target) }),
            32 => unsafe { eq_mask_32(self.as_ptr(), target) },
            _ => panic!("group matching requires 16 or 32 byte chunks"),
        }
    }
}

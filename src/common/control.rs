use super::layout::GROUP_SIZE;
use super::simd::{eq_mask_16, eq_mask_32, free_mask_16, free_mask_32, preferred_group_width};

pub(crate) const CTRL_EMPTY: u8 = 0;
pub(crate) const CTRL_TOMBSTONE: u8 = 0x80;

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
    let low = (hash as u8) & 0x7F;
    low.max(1)
}

#[inline]
pub(crate) fn fingerprint_bit(fingerprint: u8) -> u128 {
    1u128 << (fingerprint.saturating_sub(1) as u32)
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
        if self.len() < GROUP_SIZE {
            return self.iter().position(|control| control.is_free());
        }

        let mut index = 0usize;
        let wide = preferred_group_width();
        while wide > GROUP_SIZE && index + wide <= self.len() {
            let mask = self[index..index + wide].match_free_group();
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += wide;
        }
        while index + GROUP_SIZE <= self.len() {
            let mask = self[index..index + GROUP_SIZE].match_free_group();
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += GROUP_SIZE;
        }

        for (offset, control) in self[index..].iter().enumerate() {
            if control.is_free() {
                return Some(index + offset);
            }
        }

        None
    }

    #[inline]
    fn find_first(&self, target: u8) -> Option<usize> {
        self.find_next(target, 0)
    }

    #[inline]
    fn find_next(&self, target: u8, start: usize) -> Option<usize> {
        if start >= self.len() {
            return None;
        }

        if self.len() - start < GROUP_SIZE {
            for (offset, &control) in self[start..].iter().enumerate() {
                if control == target {
                    return Some(start + offset);
                }
            }
            return None;
        }

        let mut index = start;
        let wide = preferred_group_width();
        while wide > GROUP_SIZE && index + wide <= self.len() {
            let mask = self[index..index + wide].match_fingerprint_group(target);
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += wide;
        }
        while index + GROUP_SIZE <= self.len() {
            let mask = self[index..index + GROUP_SIZE].match_fingerprint_group(target);
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            index += GROUP_SIZE;
        }

        for (offset, &control) in self[index..].iter().enumerate() {
            if control == target {
                return Some(index + offset);
            }
        }

        None
    }

    #[inline]
    fn match_free_group(&self) -> u32 {
        match self.len() {
            GROUP_SIZE => free_mask_16(self.as_ptr()) as u32,
            32 => free_mask_32(self.as_ptr()),
            _ => panic!("group matching requires 16 or 32 byte chunks"),
        }
    }

    #[inline]
    fn match_fingerprint_group(&self, target: u8) -> u32 {
        match self.len() {
            GROUP_SIZE => eq_mask_16(self.as_ptr(), target) as u32,
            32 => eq_mask_32(self.as_ptr(), target),
            _ => panic!("group matching requires 16 or 32 byte chunks"),
        }
    }
}

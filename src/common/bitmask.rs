//! Architecture-abstracted bitmask over a control group.
//!
//! On `x86_64`, the mask is a `u16` where bit `i` indicates slot `i`.
//! On aarch64, the mask is a `u64` where nibble `i` (4 bits) indicates slot `i`
//! — this is the native representation produced by `vshrn_n_u16`, which is
//! cheaper than synthesizing a 1-bit-per-byte movemask via `vaddv_u8`.
//!
//! Callers use `BitMask` via its iterator and helper methods; the underlying
//! representation is hidden.

#[cfg(target_arch = "aarch64")]
pub(crate) type BitMaskWord = u64;
#[cfg(not(target_arch = "aarch64"))]
pub(crate) type BitMaskWord = u16;

#[cfg(target_arch = "aarch64")]
pub(crate) const BITMASK_STRIDE: u32 = 4;
#[cfg(not(target_arch = "aarch64"))]
pub(crate) const BITMASK_STRIDE: u32 = 1;

pub(crate) struct BitMask(pub(crate) BitMaskWord);

impl BitMask {
    #[inline]
    /// True if any slot is set.
    pub(crate) fn any(self) -> bool {
        self.0 != 0
    }

    /// Index of the lowest set slot, or `None` if empty.
    #[inline]
    pub(crate) fn lowest(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some((self.0.trailing_zeros() / BITMASK_STRIDE) as usize)
        }
    }

    /// Restrict the mask to the first `n` slots. Slots `>= n` are cleared.
    #[inline]
    pub(crate) fn truncate_to(self, n: usize) -> Self {
        // A full group is GROUP_SIZE=16 slots. If n >= 16, no truncation.
        if n >= 16 {
            return self;
        }
        #[allow(clippy::cast_possible_truncation)]
        let bits = (n as u32) * BITMASK_STRIDE;
        let mask = (1 as BitMaskWord).wrapping_shl(bits).wrapping_sub(1);
        Self(self.0 & mask)
    }
}

// `BitMask` is consumed by iteration (bits cleared as yielded). Copy is used
// intentionally so callers can pass/store masks by value without worrying
// about ownership. Consuming iteration on a Copy type is safe because the
// mask snapshot is frozen at call site.
#[allow(clippy::copy_iterator)]
impl Iterator for BitMask {
    type Item = usize;

    #[inline]
    /// Yields the index of the lowest set slot, then clears it, until empty.
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros();
        let slot = (bit / BITMASK_STRIDE) as usize;
        // Clear all bits for this slot (one bit on x86, full nibble on aarch64).
        #[cfg(target_arch = "aarch64")]
        {
            let nibble = (0xFu64).wrapping_shl(bit);
            self.0 &= !nibble;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0 &= self.0.wrapping_sub(1);
        }
        Some(slot)
    }
}

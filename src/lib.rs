mod common;
mod elastic;
mod funnel;

pub use elastic::ElasticHashMap;
pub use funnel::FunnelHashMap;

#[doc(hidden)]
pub mod bench_support {
    #[inline]
    pub fn control_match_free_group(chunk: &[u8]) -> u32 {
        <[u8] as crate::common::Controls>::match_free_group(chunk)
    }

    #[inline]
    pub fn control_match_fingerprint_group(chunk: &[u8], target: u8) -> u32 {
        <[u8] as crate::common::Controls>::match_fingerprint_group(chunk, target)
    }

    pub const CTRL_EMPTY: u8 = crate::common::CTRL_EMPTY;
}

mod common;
mod elastic;
mod funnel;

pub use elastic::ElasticHashMap;
pub use elastic::ElasticOptions;
pub use funnel::FunnelHashMap;
pub use funnel::FunnelOptions;

#[doc(hidden)]
pub mod bench_support {
    #[inline]
    #[must_use]
    pub fn control_match_free_group(chunk: &[u8]) -> u32 {
        <[u8] as crate::common::control::Controls>::match_free_group(chunk)
    }

    #[inline]
    #[must_use]
    pub fn control_match_fingerprint_group(chunk: &[u8], target: u8) -> u32 {
        <[u8] as crate::common::control::Controls>::match_fingerprint_group(chunk, target)
    }

    pub const CTRL_EMPTY: u8 = crate::common::control::CTRL_EMPTY;
}

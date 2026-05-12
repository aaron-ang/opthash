// When the `mini-hash` feature is off, `mini_hash(_)` returns `()` and the
// per-slot mini-hash variables threaded through the lookup paths become
// unit-valued. The bindings are still meaningful for readability and
// compile out completely; silence the pedantic lint instead of polluting
// every call site with `#[cfg]`.
#![cfg_attr(not(feature = "mini-hash"), allow(clippy::let_unit_value))]

mod common;
mod elastic;
mod funnel;

#[cfg(feature = "python")]
mod python;

pub use elastic::ElasticHashMap;
pub use elastic::ElasticOptions;
pub use funnel::FunnelHashMap;
pub use funnel::FunnelOptions;

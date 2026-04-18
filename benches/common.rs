#![allow(dead_code, clippy::must_use_candidate)]

use std::collections::HashMap as StdHashMap;

use opthash::{ElasticHashMap, FunnelHashMap};

pub const LATENCY_SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];

pub fn key_at(index: usize) -> u64 {
    (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

pub fn make_pairs(count: usize) -> Vec<(u64, u64)> {
    (0..count)
        .map(|idx| {
            let key = key_at(idx);
            (key, key ^ 0xA5A5_A5A5_A5A5_A5A5)
        })
        .collect()
}

pub fn build_std_map(pairs: &[(u64, u64)]) -> StdHashMap<u64, u64> {
    let mut map = StdHashMap::with_capacity(pairs.len() * 2);
    for &(key, value) in pairs {
        map.insert(key, value);
    }
    map
}

pub fn build_elastic_map(pairs: &[(u64, u64)]) -> ElasticHashMap<u64, u64> {
    let mut map = ElasticHashMap::with_capacity(pairs.len() * 2);
    for &(key, value) in pairs {
        map.insert(key, value);
    }
    map
}

pub fn build_funnel_map(pairs: &[(u64, u64)]) -> FunnelHashMap<u64, u64> {
    let mut map = FunnelHashMap::with_capacity(pairs.len() * 2);
    for &(key, value) in pairs {
        map.insert(key, value);
    }
    map
}

pub fn size_label(size: usize) -> String {
    if size >= 1_000_000 {
        format!("{}M", size / 1_000_000)
    } else if size >= 1_000 {
        format!("{}K", size / 1_000)
    } else {
        format!("{size}")
    }
}

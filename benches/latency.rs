#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod common;

use std::collections::HashMap as StdHashMap;
use std::env;
use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use common::{
    GOLDEN_RATIO_U64, VALUE_XOR_MIX, build_elastic_map, build_funnel_map, build_std_map, key_at,
    make_pairs,
};
use hdrhistogram::Histogram;
use opthash::{ElasticHashMap, FunnelHashMap};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Op {
    GetHit,
    GetMiss,
    Insert,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::GetHit => "get-hit",
            Op::GetMiss => "get-miss",
            Op::Insert => "insert",
        }
    }
    fn parse(s: &str) -> Result<Op, String> {
        match s {
            "get-hit" => Ok(Op::GetHit),
            "get-miss" => Ok(Op::GetMiss),
            "insert" => Ok(Op::Insert),
            other => Err(format!("unknown op '{other}'")),
        }
    }
}

const ALL_MAPS: &[&str] = &["std", "elastic", "funnel"];
const DEFAULT_OPS: &[Op] = &[Op::GetHit, Op::GetMiss, Op::Insert];
const DEFAULT_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];
const DEFAULT_SAMPLES: usize = 1_000_000;
const DEFAULT_WARMUP: usize = 10_000;

struct Args {
    sizes: Vec<usize>,
    ops: Vec<Op>,
    maps: Vec<&'static str>,
    samples: usize,
    warmup: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        sizes: DEFAULT_SIZES.to_vec(),
        ops: DEFAULT_OPS.to_vec(),
        maps: ALL_MAPS.to_vec(),
        samples: DEFAULT_SAMPLES,
        warmup: DEFAULT_WARMUP,
    };
    let mut it = env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--size" => {
                let v = it.next().expect("--size needs value");
                a.sizes = v
                    .split(',')
                    .map(|s| s.parse().expect("size must be usize"))
                    .collect();
            }
            "--op" => {
                let v = it.next().expect("--op needs value");
                a.ops = v.split(',').map(|s| Op::parse(s).unwrap()).collect();
            }
            "--map" => {
                let v = it.next().expect("--map needs value");
                a.maps = v
                    .split(',')
                    .map(|s| match s {
                        "std" => "std",
                        "elastic" => "elastic",
                        "funnel" => "funnel",
                        other => panic!("unknown map '{other}'"),
                    })
                    .collect();
            }
            "--samples" => {
                a.samples = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--samples int");
            }
            "--warmup" => {
                a.warmup = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--warmup int");
            }
            _ if flag.starts_with("--") => eprintln!("warning: ignoring flag {flag}"),
            _ => {}
        }
    }
    a
}

fn measure_clock_overhead_ns() -> u64 {
    let n = 10_000u64;
    let t0 = Instant::now();
    for _ in 0..n {
        black_box(Instant::now());
    }
    (t0.elapsed().as_nanos() as u64) / n
}

fn new_hist() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 10_000_000, 3).expect("valid hdr bounds")
}

fn scatter(i: usize, n: usize) -> usize {
    ((i as u64).wrapping_mul(GOLDEN_RATIO_U64) as usize) % n
}

fn measure<F, R>(samples: usize, warmup: usize, mut op: F) -> Histogram<u64>
where
    F: FnMut(usize) -> R,
{
    for i in 0..warmup {
        black_box(op(i));
    }
    let mut h = new_hist();
    for i in 0..samples {
        let t0 = Instant::now();
        let r = op(i);
        let dt = t0.elapsed().as_nanos() as u64;
        black_box(r);
        h.record(dt.max(1)).unwrap();
    }
    h
}

fn run_get_hit(map: &str, size: usize, samples: usize, warmup: usize) -> Histogram<u64> {
    let pairs = make_pairs(size);
    let keys: Vec<u64> = pairs.iter().map(|&(k, _)| k).collect();
    let n = keys.len();
    match map {
        "std" => {
            let m = build_std_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&keys[scatter(i, n)])).copied()
            })
        }
        "elastic" => {
            let m = build_elastic_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&keys[scatter(i, n)])).copied()
            })
        }
        "funnel" => {
            let m = build_funnel_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&keys[scatter(i, n)])).copied()
            })
        }
        _ => unreachable!(),
    }
}

fn run_get_miss(map: &str, size: usize, samples: usize, warmup: usize) -> Histogram<u64> {
    let pairs = make_pairs(size);
    let miss_keys: Vec<u64> = (0..size).map(|i| key_at(i + 100_000_000)).collect();
    let n = miss_keys.len();
    match map {
        "std" => {
            let m = build_std_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&miss_keys[scatter(i, n)])).copied()
            })
        }
        "elastic" => {
            let m = build_elastic_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&miss_keys[scatter(i, n)])).copied()
            })
        }
        "funnel" => {
            let m = build_funnel_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&miss_keys[scatter(i, n)])).copied()
            })
        }
        _ => unreachable!(),
    }
}

fn run_insert(map: &str, samples: usize, warmup: usize) -> Histogram<u64> {
    let total = samples + warmup;
    let insert_keys: Vec<(u64, u64)> = (0..total)
        .map(|i| {
            let k = key_at(i + 200_000_000);
            (k, k ^ VALUE_XOR_MIX)
        })
        .collect();
    match map {
        "std" => {
            let mut m: StdHashMap<u64, u64> = StdHashMap::new();
            measure(samples, warmup, |i| {
                let (k, v) = insert_keys[i];
                m.insert(k, v)
            })
        }
        "elastic" => {
            let mut m: ElasticHashMap<u64, u64> = ElasticHashMap::new();
            measure(samples, warmup, |i| {
                let (k, v) = insert_keys[i];
                m.insert(k, v)
            })
        }
        "funnel" => {
            let mut m: FunnelHashMap<u64, u64> = FunnelHashMap::new();
            measure(samples, warmup, |i| {
                let (k, v) = insert_keys[i];
                m.insert(k, v)
            })
        }
        _ => unreachable!(),
    }
}

fn run(map: &str, size: usize, op: Op, samples: usize, warmup: usize) -> Histogram<u64> {
    match op {
        Op::GetHit => run_get_hit(map, size, samples, warmup),
        Op::GetMiss => run_get_miss(map, size, samples, warmup),
        Op::Insert => run_insert(map, samples, warmup),
    }
}

fn write_json(
    map: &str,
    size: usize,
    op: Op,
    h: &Histogram<u64>,
    overhead: u64,
    samples: usize,
) -> std::io::Result<PathBuf> {
    let dir = PathBuf::from(format!("target/latency/{map}/{size}"));
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", op.name()));
    let mut f = fs::File::create(&path)?;

    let p50 = h.value_at_quantile(0.50);
    let p90 = h.value_at_quantile(0.90);
    let p99 = h.value_at_quantile(0.99);
    let p999 = h.value_at_quantile(0.999);
    let p9999 = h.value_at_quantile(0.9999);
    let p99999 = h.value_at_quantile(0.99999);
    let max = h.max();
    let mean = h.mean();
    let cap = p99999.saturating_mul(2).max(p9999.saturating_mul(4));

    writeln!(f, "{{")?;
    writeln!(f, "  \"map\": \"{map}\",")?;
    writeln!(f, "  \"size\": {size},")?;
    writeln!(f, "  \"op\": \"{}\",", op.name())?;
    writeln!(f, "  \"samples\": {samples},")?;
    writeln!(f, "  \"clock_overhead_ns\": {overhead},")?;
    writeln!(f, "  \"percentiles\": {{")?;
    writeln!(f, "    \"p50\": {p50},")?;
    writeln!(f, "    \"p90\": {p90},")?;
    writeln!(f, "    \"p99\": {p99},")?;
    writeln!(f, "    \"p999\": {p999},")?;
    writeln!(f, "    \"p9999\": {p9999},")?;
    writeln!(f, "    \"p99999\": {p99999},")?;
    writeln!(f, "    \"max\": {max},")?;
    writeln!(f, "    \"mean\": {mean:.3}")?;
    writeln!(f, "  }},")?;
    write!(f, "  \"histogram\": [")?;
    let mut first = true;
    for v in h.iter_recorded() {
        let hi = v.value_iterated_to();
        if hi > cap {
            break;
        }
        let count = v.count_since_last_iteration();
        if count == 0 {
            continue;
        }
        let lo = h.lowest_equivalent(hi);
        if first {
            writeln!(f)?;
        } else {
            writeln!(f, ",")?;
        }
        first = false;
        write!(
            f,
            "    {{\"ns_low\": {lo}, \"ns_high\": {hi}, \"count\": {count}}}"
        )?;
    }
    if !first {
        writeln!(f)?;
        write!(f, "  ")?;
    }
    writeln!(f, "]")?;
    writeln!(f, "}}")?;
    Ok(path)
}

fn main() {
    let args = parse_args();
    let overhead = measure_clock_overhead_ns();
    eprintln!("clock_overhead_ns ≈ {overhead}");

    for &size in &args.sizes {
        for &op in &args.ops {
            for &m in &args.maps {
                if op == Op::Insert && size != args.sizes[0] {
                    // Insert ignores `size` (uses `samples`), so run once per map instead of per size.
                    continue;
                }
                eprint!(
                    "running map={m} size={size} op={} samples={} ... ",
                    op.name(),
                    args.samples
                );
                let t0 = Instant::now();
                let h = run(m, size, op, args.samples, args.warmup);
                let dur = t0.elapsed();
                let path = write_json(m, size, op, &h, overhead, args.samples)
                    .expect("write latency json");
                eprintln!(
                    "done in {:.1}s | p50={}ns p99={}ns p999={}ns max={}ns → {}",
                    dur.as_secs_f64(),
                    h.value_at_quantile(0.50),
                    h.value_at_quantile(0.99),
                    h.value_at_quantile(0.999),
                    h.max(),
                    path.display()
                );
            }
        }
    }
}
